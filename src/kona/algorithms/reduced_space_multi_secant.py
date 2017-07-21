from kona.algorithms.base_algorithm import OptimizationAlgorithm

class ReducedSpaceMultiSecant(OptimizationAlgorithm):
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
        super(ReducedSpaceMultiSecant, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        # PrimalDualVectors = X, dX, R, dLdX, obj_grad, dX_safe
        num_pd = 6
        self.primal_factory.request_num_vectors(num_pd)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(num_pd)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(num_pd)
        self.state_factory.request_num_vectors(3)

        # iteration counter
        self.iter = 0

        # set the preconditioner for the multi-secant methods
        self.precond = get_opt(self.optns, None, 'multi_secant', 'precond')
        if self.precond is None:
            # use the identity preconditioner
            self.precond = IdentityMatrix()
        elif self.precond is 'approx_schur':
            # use the SVD-based approximate Schur preconditioner
            precond_optns = get_opt(self.optns, {}, 'multi_secant')
            self.precond = ApproxSchur([self.primal_factory, self.state_factory, self.eq_factory,
                                        self.ineq_factory], precond_optns)
        elif self.precond is 'idf_schur':
            self.precond = ReducedSchurPreconditioner(
                [primal_factory, state_factory, eq_factory, ineq_factory])
        else:
            raise BadKonaOption(self.optns, 'precond')

        # set the type of multi-secant method
        try:
            # the multisecant member is for the "second"-order scheme
            multisecant = get_opt(
                self.optns, AndersonMultiSecant, 'multi_secant', 'type')
            hessian_optns = get_opt(self.optns, {}, 'multi_secant')
            hessian_optns['out_file'] = self.info_file
            self.hess_reg = get_opt(self.optns, 0.0, 'multi_secant', 'hess_reg')
            self.multisecant = multisecant([self.primal_factory, self.eq_factory,
                                            self.ineq_factory], hessian_optns)
            # the multisecant_safe member is for the first-order scheme
            self.multisecant_safe = multisecant([self.primal_factory, self.eq_factory,
                                                 self.ineq_factory], hessian_optns)

        except Exception:
            raise BadKonaOption(self.optns, 'multi_secant','type')

        # set remaining options
        self.primal_tol_abs = get_opt(self.optns, 1e-6, 'opt_tol_abs')
        self.cnstr_tol_abs = get_opt(self.optns, 1e-6, 'feas_tol_abs')
        self.beta = get_opt(self.optns, 1.0, 'multi_secant', 'beta')

        # TEMPORARY!!!
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
            '#  iter' + ' '*5 +
            '   cost' + ' '*5 +
            'optimality  ' + ' '*5 +
            'feasibility ' + ' '*5 +
            'objective   ' + '\n'
        )

    def _write_history(self, opt, feas, obj, info):
        self.hist_file.write(
            '%7i'%self.iter + ' '*5 +
            '%7i'%self.primal_factory._memory.cost + ' '*5 +
            '%11e'%opt + ' '*5 +
            '%11e'%feas + ' '*5 +
            '%11e'%obj + ' '*5 +
            info + '\n'
        )

    def _generate_vector(self):
        """
        Create appropriate vector based on vector factory
        """
        assert self.primal_factory is not None, \
            'ReducedSpaceMultiSecant() >> primal_factory is not defined!'
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
            '***        Using Multi-Secant Algorithm        ***\n' +
            '**************************************************\n' +
            '\n')

        # generate primal-dual vectors, and other vectors
        X = self._generate_vector()
        dX = self._generate_vector()
        dX_safe = self._generate_vector()
        R = self._generate_vector()
        dLdX = self._generate_vector()
        obj_grad = self._generate_vector()

        # generate state vectors
        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        adjoint = self.state_factory.generate()

        # evaluate the initial design, state, and adjoint before starting outer iterations
        X.equals_init_guess()
        state.equals_primal_solution(X.primal)
        obj_val = objective_value(X.primal, state)

        # initialize the multi-secant methods
        self.multisecant.set_initial_data(X)
        self.multisecant_safe.set_initial_data(X)

        # initialize the preconditioner
        self.precond.linearize(X.primal, state)

        # get the adjoint for -h^T\lambda_h - g^T\lambda_g
        adjoint.equals_homotopy_adjoint(X, state, state_work, obj_scale=0.0)
        R.equals_KKT_conditions(X, state, adjoint, obj_scale=0.0)
        dLdX.equals(R)
        R.primal.plus(X.primal)
        self.multisecant_safe.add_to_history(X, R)

        # get the adjoint for df/dx, then add it to finish dLdX computation
        adjoint.equals_objective_adjoint(X.primal, state, state_work)
        obj_grad.equals(0.)
        obj_grad.primal.equals_total_gradient(X.primal, state, adjoint)
        dLdX.primal.plus(obj_grad.primal)
        self.multisecant.add_to_history(X, dLdX)

        # send initial point info to the user
        solver_info = current_solution(self.iter, X.primal, state, adjoint, X.get_dual())
        if isinstance(solver_info, str):
            self.info_file.write('\n' + solver_info + '\n')

        # get initial optimality and feasibility
        R.equals_primaldual_residual(dLdX, X.ineq)
        opt, feas = R.get_optimality_and_feasiblity()
        grad_norm0 = max(opt, EPS)
        feas_norm0 = max(feas, EPS)
        self.info_file.write(
            'grad_norm0         = %e\n'%grad_norm0 +
            'feas_norm0         = %e\n'%feas_norm0)

        # Begin iterations
        converged = False
        info = ' '
        for i in xrange(self.max_iter):
            self.iter += 1

            # evaluate optimality and feasibility, and output as necessary
            # Note: dLdX should be up-to-date at this point
            R.equals_primaldual_residual(dLdX, X.ineq)
            grad_norm, feas_norm = R.get_optimality_and_feasiblity()
            self.info_file.write(
                '==========================================================\n' +
                'Iteration %i\n'%self.iter)
            self.info_file.write(
                'grad_norm          = %e (%e <-- tolerance)\n'%(
                    grad_norm, self.primal_tol) +
                'feas_norm          = %e (%e <-- tolerance)\n'%(
                    feas_norm, self.cnstr_tol))
            # write convergence history
            self._write_history(grad_norm, feas_norm, obj_val, info)

            # check for convergence
            if grad_norm < self.primal_tol*grad_norm0 + self.primal_tol_abs and \
               feas_norm < self.cnstr_tol*feas_norm0 + self.cnstr_tol_abs:
                converged = True
                break

            # get full multi-secant step and projected-gradient
            self.multisecant.build_difference_matrices(self.hess_reg)
            self.multisecant.solve(R, dX, self.beta, self.precond.product)
            self.multisecant_safe.build_difference_matrices(self.hess_reg)
            self.multisecant_safe.solve(obj_grad, dX_safe, precond=self.precond.product)
            info = ' '
            if False: #dX.primal.inner(dX_safe.primal) < 0.:
                # the full step is considered unsafe: compute first-order step
                self.multisecant_safe.solve(R, dX, self.beta, self.precond.product)
                info += 'safe-step'
                # remove history from multisecant
                #self.multisecant.clear_history()

            # safe-guard against large steps
            if dX.norm2 > self.radius_max: #/self.iter:
                dX.times(self.radius_max/(dX.norm2)) #*self.iter))
                info += ' length restricted'
            X.plus(dX)

            # j = 0
            # while True:
            #     j += 1
            #     X.plus(dX)
            #     # evaluate at new X, and check if optimality improved
            #     state.equals_primal_solution(X.primal)
            #     adjoint.equals_homotopy_adjoint(X, state, state_work)
            #     dLdX.equals_KKT_conditions(X, state, adjoint)
            #     R.equals_primaldual_residual(dLdX, X.ineq)
            #     self.info_file.write(
            #         'line search iteration %i: |R(x_k)| = %e: |R(x_k + p_k)| = %e\n'%(
            #             j, grad_norm**2 + feas_norm**2, R.norm2**2))
            #     if R.norm2**2 < grad_norm**2 + feas_norm**2 or j >= 4:
            #         break
            #     X.minus(dX)
            #     dX.times(0.1)

            # evaluate at new X, construct first-order optimality conditions, and store
            state.equals_primal_solution(X.primal)
            obj_val = objective_value(X.primal, state)

            self.precond.linearize(X.primal, state)

            # get the adjoint for -h^T\lambda_h - g^T\lambda_g
            adjoint.equals_homotopy_adjoint(X, state, state_work, obj_scale=0.0)
            R.equals_KKT_conditions(X, state, adjoint, obj_scale=0.0)
            dLdX.equals(R)
            R.primal.plus(X.primal)
            self.multisecant_safe.add_to_history(X, R)

            # get the adjoint for df/dx, then add it to finish dLdX computation
            adjoint.equals_objective_adjoint(X.primal, state, state_work)
            obj_grad.primal.equals_total_gradient(X.primal, state, adjoint)
            dLdX.primal.plus(obj_grad.primal)
            self.multisecant.add_to_history(X, dLdX)

            # output current solution info to the user
            solver_info = current_solution(self.iter, X.primal, state, adjoint, X.get_dual())
            if isinstance(solver_info, str):
                self.info_file.write('\n' + solver_info + '\n')

        # end of "Newton" iterations
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
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.preconds.schur import ApproxSchur
from kona.linalg.matrices.preconds import ReducedSchurPreconditioner
from kona.linalg.solvers.util import EPS
