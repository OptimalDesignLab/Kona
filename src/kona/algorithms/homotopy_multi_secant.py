from kona.algorithms.base_algorithm import OptimizationAlgorithm


class HomotopyMultiSecant(OptimizationAlgorithm):
    """
    Algorithm for generic nonlinearly constrained optiimzation problems.

    Parameters
    ----------
    primal_factory : VectorFactory
    state_factory : VectorFactory
    eq_factory : VectorFactory
    ineq_factory: VectorFactory
    optns : dict, optional
    """

    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(HomotopyMultiSecant, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        # PrimalDualVectors = X, X_init, dX, R, dLdX, dXdmu, dXdmu_old
        num_pd = 7
        self.primal_factory.request_num_vectors(num_pd)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(num_pd)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(num_pd)
        self.state_factory.request_num_vectors(3)

        # iteration counters
        self.iter = 0
        self.inner = 0

        # set the type of multi-secant method
        try:
            multisecant = get_opt(
                self.optns, AndersonMultiSecant, 'multi_secant', 'type')
            hessian_optns = get_opt(self.optns, {}, 'multi_secant')
            hessian_optns['out_file'] = self.info_file
            self.multisecant = multisecant([self.primal_factory, self.eq_factory,
                                            self.ineq_factory], hessian_optns)
        except Exception:
            raise BadKonaOption(self.optns, 'multi_secant','type')

        # homotopy options
        self.mu = get_opt(self.optns, 0.0, 'homotopy', 'mu_init')
        self.inner_grad_tol = get_opt(self.optns, 1e-2, 'homotopy', 'inner_grad_tol')
        self.inner_feas_tol = get_opt(self.optns, 1e-2, 'homotopy', 'inner_feas_tol')
        self.inner_max_iter = get_opt(self.optns, 10, 'homotopy', 'inner_maxiter')
        self.dmu = get_opt(self.optns, 0.1, 'homotopy', 'init_step')
        self.target_angle = get_opt(self.optns, 5.0*np.pi/180., 'homotopy', 'nominal_angle')
        self.dmu_max = get_opt(self.optns, 0.1, 'homotopy', 'max_step')
        self.dmu_min = get_opt(self.optns, 0.001, 'homotopy', 'min_step')
        self.radius_max = get_opt(self.optns, 1.0, 'homotopy', 'radius_max')

        # The following data members are set by super class
        # self.primal_tol
        # self.cnstr_tol
        # self.max_iter
        # self.info_file
        # self.hist_file

    def _write_header(self):
        self.hist_file.write(
            '# Kona homotopy-globalized multi-secant convergence history file\n' +
            '# outer' + ' '*5 +
            '  inner' + ' '*5 +
            '   cost' + ' '*5 +
            'optimality  ' + ' '*5 +
            'feasibility ' + ' '*5 +
            'objective   ' + ' '*5 +
            'mu param    ' + '\n'
        )

    def _write_history(self, opt, feas, obj):
        self.hist_file.write(
            '%7i'%self.iter + ' '*5 +
            '%7i'%self.inner + ' '*5 +
            '%7i'%self.primal_factory._memory.cost + ' '*5 +
            '%11e'%opt + ' '*5 +
            '%11e'%feas + ' '*5 +
            '%11e'%obj + ' '*5 +
            '%11e'%self.mu + '\n'
        )

    def _generate_vector(self):
        """
        Create appropriate vector based on vector factory
        """
        assert self.primal_factory is not None, \
            'HomotopyMultiSecant() >> primal_factory is not defined!'
        dual_eq = None
        dual_ineq = None
        primal = self.primal_factory.generate()
        if self.eq_factory is not None:
            dual_eq = self.eq_factory.generate()
        if self.ineq_factory is not None:
            dual_ineq = self.ineq_factory.generate()
        return PrimalDualVector(primal, eq_vec=dual_eq, ineq_vec=dual_ineq)

    def solve(self):
        self._write_header()
        self.info_file.write(
            '\n' +
            '**************************************************\n' +
            '***        Using Multi-Secant Homotopy         ***\n' +
            '**************************************************\n' +
            '\n')

        # generate primal-dual vectors
        X = self._generate_vector()
        X_init = self._generate_vector()
        dX = self._generate_vector()
        R = self._generate_vector()
        dLdX = self._generate_vector()
        dXdmu = self._generate_vector()
        dXdmu_old = self._generate_vector()

        # generate state vectors
        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        adjoint = self.state_factory.generate()

        # evaluate the initial design, state, and adjoint before starting outer iterations
        X.equals_init_guess()
        X_init.equals(X)
        state.equals_primal_solution(X.primal)
        adjoint.equals_homotopy_adjoint(X, state, state_work)

        # send initial point info to the user
        solver_info = current_solution(self.iter, X.primal, state, adjoint, X.get_dual())
        if isinstance(solver_info, str):
            self.info_file.write('\n' + solver_info + '\n')

        # initialize the multi-secant method
        dLdX.equals_KKT_conditions(X, state, adjoint)
        self.multisecant.add_to_history(X, dLdX)
        dXdmu.equals(dLdX)
        dXdmu.primal.equals(X_init.primal)
        self.multisecant.set_initial_data(dXdmu)

        # initialize the tangent vectors
        dXdmu.equals(0.)
        dXdmu_old.equals(0.)

        # get initial optimality and feasibility
        R.equals_homotopy_residual(dLdX, X, X_init, mu=1.0)
        opt, feas = R.get_optimality_and_feasiblity()
        grad_norm0 = max(opt, EPS)
        feas_norm0 = max(feas, EPS)
        self.info_file.write(
            'grad_norm0         = %e\n'%grad_norm0 +
            'feas_norm0         = %e\n'%feas_norm0)

        # Begin outer (homotopy) loop here
        converged = False
        for i in range(self.max_iter):
            self.iter += 1
            self.inner = 0

            # evaluate optimality and feasibility, and output as necessary
            # Note: dLdX should be up-to-date at this point
            R.equals_homotopy_residual(dLdX, X, X_init, mu=1.0)
            grad_norm, feas_norm = R.get_optimality_and_feasiblity()
            self.info_file.write(
                '==========================================================\n' +
                'Outer (Homotopy) Iteration %i: mu = %e\n\n'%(self.iter,self.mu))

            self.info_file.write(
                'grad_norm          = %e (%e <-- tolerance)\n'%(
                    grad_norm, self.primal_tol) +
                'feas_norm          = %e (%e <-- tolerance)\n'%(
                    feas_norm, self.cnstr_tol))

            # write convergence history
            obj_val = objective_value(X.primal, state)
            self._write_history(grad_norm, feas_norm, obj_val)

            # check for convergence
            if grad_norm < self.primal_tol*grad_norm0 and \
               feas_norm < self.cnstr_tol*feas_norm0:
                converged = True
                break

            # find the predictor step
            R.equals_predictor_rhs(dLdX, X, X_init, mu=self.mu)
            self.multisecant.build_difference_matrices_for_homotopy(mu=self.mu)
            dXdmu_old.equals(dXdmu)
            self.multisecant.solve(R, dXdmu)
            # adjust the step size
            if self.iter > 1:
                denom = np.sqrt(dXdmu.norm2**2 + 1)*np.sqrt(dXdmu_old.norm2**2 + 1)
                numer = (dXdmu.inner(dXdmu_old) + 1.)
                phi = math.acos(max(-1., min(1., numer/denom)))
                if phi > EPS:
                    self.dmu /= (phi/self.target_angle)
            self.dmu = min(min(self.dmu_max, self.radius_max/dXdmu.norm2), self.dmu)
            self.dmu = max(self.dmu_min, self.dmu)
            # take the predictor step
            X.equals_ax_p_by(1.0, X, self.dmu, dXdmu)
            self.mu = min(self.mu + self.dmu, 1.0)

            # evaluate the optimality residual and add to multisecant
            state.equals_primal_solution(X.primal)
            adjoint.equals_homotopy_adjoint(X, state, state_work)
            dLdX.equals_KKT_conditions(X, state, adjoint)
            self.multisecant.add_to_history(X, dLdX)

            R.equals_homotopy_residual(dLdX, X, X_init, mu=self.mu)
            opt, feas = R.get_optimality_and_feasiblity()
            inner_grad_norm0 = opt
            inner_feas_norm0 = max(feas, EPS)
            inner_grad_norm = opt
            inner_feas_norm = feas
            self.info_file.write(
                '-------------------- Corrector Iterations\n' +
                'inner_grad_norm0 = %e'%inner_grad_norm0 +
                ': inner_feas_norm0 = %e\n'%inner_feas_norm0)

            # inner (corrector) iterations
            for k in range(self.inner_max_iter):
                self.inner += 1
                self.info_file.write(
                    'Corr. Iter. %i'%self.inner +
                    ': inner_grad_norm = %e'%inner_grad_norm +
                    ': inner_feas_norm = %e\n'%inner_feas_norm)

                # check for convergence
                if np.fabs(self.mu - 1.0) < EPS:
                    # check for outer convergence if mu = 1.0
                    if grad_norm < self.primal_tol*grad_norm0 and \
                       feas_norm < self.cnstr_tol*feas_norm0:
                        converged = True
                        break
                else: # check for inner convergence
                    if inner_grad_norm < self.inner_grad_tol*inner_grad_norm0 and \
                       inner_feas_norm < self.inner_feas_tol*inner_feas_norm0:
                        break

                # solve for corrector step
                self.multisecant.build_difference_matrices_for_homotopy(mu=self.mu)
                self.multisecant.solve(R, dX)
                # safe-guard against large steps
                if dX.norm2 > self.radius_max:
                    dX.times(self.radius_max/dX.norm2)
                X.plus(dX)

                # evaluate at new X and store data
                state.equals_primal_solution(X.primal)
                obj_val = objective_value(X.primal, state)
                adjoint.equals_homotopy_adjoint(X, state, state_work)
                dLdX.equals_KKT_conditions(X, state, adjoint)
                self.multisecant.add_to_history(X, dLdX)
                R.equals_homotopy_residual(dLdX, X, X_init, mu=1.0)
                grad_norm, feas_norm = R.get_optimality_and_feasiblity()
                R.equals_homotopy_residual(dLdX, X, X_init, mu=self.mu)
                inner_grad_norm, inner_feas_norm = R.get_optimality_and_feasiblity()
                # write convergence history
                self._write_history(grad_norm, feas_norm, obj_val)

            # end of inner (corrector) iterations

            # send initial point info to the user
            solver_info = current_solution(self.iter, X.primal, state, adjoint, X.get_dual())
            if isinstance(solver_info, str):
                self.info_file.write('\n' + solver_info + '\n')

        # end of outer (homotopy) iterations
        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Optimization FAILED!\n')
        self.info_file.write(
            'Total number of nonlinear iterations: %i\n\n'%self.iter)

# imports here to prevent circular errors
import math
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, objective_value
from kona.linalg.vectors.composite import PrimalDualVector
from kona.linalg.matrices.hessian import AndersonMultiSecant
from kona.linalg.solvers.util import EPS
