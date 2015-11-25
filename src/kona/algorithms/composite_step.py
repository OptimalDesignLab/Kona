import numpy as np

from kona.options import get_opt
from kona.linalg import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.common import dCdU, dRdU
from kona.linalg.matrices.hessian import NormalKKTMatrix, TangentKKTMatrix
from kona.linalg.solvers.util import EPS
from kona.algorithms.base_algorithm import OptimizationAlgorithm

class CompositeStep(OptimizationAlgorithm):
    """
    A reduced-space composite-step optimization algorithm for PDE-governed
    (in)equality constrained problems.

    This algorithm uses a novel 2nd order adjoint formulation for constraint
    jacobian and constrained hessian products.

    Inequality constraints are converted to equality constraints using slack
    terms of the form :math:`e^s` where `s` are the slack variables.

    Parameters
    ----------
    primal_factory : VectorFactory
    state_factory : VectorFactory
    dual_factory : VectorFactory
    optns : dict, optional
    """
    def __init__(self, primal_factory, state_factory, dual_factory, optns={}):
        # trigger base class initialization
        super(CompositeStep, self).__init__(
            primal_factory, state_factory, dual_factory, optns)

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(9)
        self.state_factory.request_num_vectors(3)
        self.dual_factory.request_num_vectors(16)

        # get general options
        self.cnstr_tol = get_opt(optns, 1e-8, 'feas_tol')
        self.factor_matrices = get_opt(optns, False, 'matrix_explicit')

        # get trust region options
        self.radius = get_opt(optns, 0.5, 'trust', 'init_radius')
        self.min_radius = get_opt(optns, 0.5/(2**3), 'trust', 'min_radius')
        self.max_radius = get_opt(optns, 0.5*(2**3), 'trust', 'max_radius')

        # get globalization type
        self.globalization = get_opt(optns, 'trust', 'globalization')

        if self.globalization == 'trust':
            # get penalty parameter options for the augmented Lagrangian
            self.mu = get_opt(optns, 0.1, 'penalty', 'mu_init')
            self.mu_pow = get_opt(optns, 0.1, 'penalty', 'mu_pow')
            self.mu_max = get_opt(optns, 1e4, 'penalty', 'mu_max')
            self.eta = 1./(self.mu**0.1)

        elif self.globalization is None:
            pass
        else:
            raise TypeError(
                'Invalid globalization! ' +
                'Can only use \'trust\'. ' +
                'If you want to skip globalization, set to None.')

        # initialize the KKT matrix definition
        normal_optns = get_opt(optns, {}, 'composite-step', 'normal-step')
        self.normal_KKT = NormalKKTMatrix(
            [self.primal_factory, self.state_factory, self.dual_factory],
            normal_optns)
        tangent_optns = get_opt(optns, {}, 'composite-step', 'tangent-step')
        self.tangent_KKT = TangentKKTMatrix(
            [self.primal_factory, self.state_factory, self.dual_factory],
            tangent_optns)
        self.tangent_KKT.set_projector(self.normal_KKT)

    def _write_header(self):
        self.hist_file.write(
            '# Kona composite-step convergence history file\n' +
            '# iters' + ' '*5 +
            '   cost' + ' '*5 +
            'optimality  ' + ' '*5 +
            'feasibility ' + ' '*5 +
            'objective   ' + ' '*5 +
            'penalty     ' + ' '*5 +
            'radius      ' + '\n'
        )

    def _write_history(self, opt, feas, obj):
        self.hist_file.write(
            '#%6i'%self.iter + ' '*5 +
            '%7i'%self.primal_factory._memory.cost + ' '*5 +
            '%11e'%opt + ' '*5 +
            '%11e'%feas + ' '*5 +
            '%11e'%obj + ' '*5 +
            '%11e'%self.mu + ' '*5 +
            '%11e'%self.radius + '\n'
        )

    def _generate_composite_primal(self):
        design = self.primal_factory.generate()
        slack = self.dual_factory.generate()
        return CompositePrimalVector(design, slack)

    def _generate_KKT_vector(self):
        primal = self._generate_composite_primal()
        dual = self.dual_factory.generate()
        return ReducedKKTVector(primal, dual)

    def solve(self):
        self._write_header()
        self.info_file.write(
            '\n' +
            '**************************************************\n' +
            '***       Using Composite-Step Algorithm       ***\n' +
            '**************************************************\n' +
            '\n')

        # generate composite KKT vectors
        X = self._generate_KKT_vector()
        P = self._generate_KKT_vector()
        dLdX = self._generate_KKT_vector()
        kkt_work = self._generate_KKT_vector()
        normal_rhs = self._generate_KKT_vector()
        normal_step = self._generate_KKT_vector()
        tangent_rhs = self._generate_composite_primal()
        tangent_step = self._generate_composite_primal()

        # generate primal vectors
        design_work = self.primal_factory.generate()

        # generate state vectors
        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        adjoint = self.state_factory.generate()

        # generate dual vectors
        dual_work = self.dual_factory.generate()
        slack_work = self.dual_factory.generate()

        # initialize basic data for outer iterations
        converged = False
        self.iter = 0

        # evaluate the initial design before starting outer iterations
        X.equals_init_guess()
        state.equals_primal_solution(X._primal._design)
        if self.factor_matrices and self.iter < self.max_iter:
            factor_linear_system(X._primal._design, state)

        # perform an adjoint solution for the Lagrangian
        state_work.equals_objective_partial(X._primal._design, state)
        dCdU(X._primal._design, state).T.product(X._dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X._primal._design, state).T.solve(state_work, adjoint)

        # send initial point info to the user
        solver_info = current_solution(
            X._primal._design, state, adjoint, X._dual, self.iter)
        if isinstance(solver_info, str):
            self.info_file.write('\n' + solver_info + '\n')

        # BEGIN NEWTON LOOP HERE
        ###############################
        min_radius_active = False
        krylov_tol = 0.00095
        for i in xrange(self.max_iter):
            # advance iteration counter
            self.iter += 1

            # evaluate optimality, feasibility and KKT norms
            dLdX.equals_KKT_conditions(
                X, state, adjoint, design_work, dual_work)
            # print info on current point
            self.info_file.write(
                '==========================================================\n' +
                'Beginning Major Iteration %i\n\n'%self.iter)
            self.info_file.write(
                'primal vars        = %e\n'%X._primal.norm2)
            self.info_file.write(
                '   design vars     = %e\n'%X._primal._design.norm2)
            self.info_file.write(
                '   slack vars      = %e\n'%X._primal._slack.norm2)
            self.info_file.write(
                'multipliers        = %e\n\n'%X._dual.norm2)

            # print 'design =', X._primal._design._data.data.x
            # print 'slack:',
            # print '   lower =', X._primal._slack._data.x_lower.x
            # print '   upper =', X._primal._slack._data.x_upper.x
            # print '   stress =', X._primal._slack._data.stress.x
            # print 'dual:',
            # print '   lower =', X._dual._data.x_lower.x
            # print '   upper =', X._dual._data.x_upper.x
            # print '   stress =', X._dual._data.stress.x

            if self.iter == 1:
                # calculate initial norms
                grad_norm0 = dLdX._primal.norm2
                design_norm0 = dLdX._primal._design.norm2
                slack_norm0 = dLdX._primal._slack.norm2
                feas_norm0 = max(dLdX._dual.norm2, EPS)
                kkt_norm0 = np.sqrt(feas_norm0**2 + grad_norm0**2)

                # set current norms to initial
                kkt_norm = kkt_norm0
                grad_norm = grad_norm0
                design_norm = design_norm0
                slack_norm = slack_norm0
                feas_norm = feas_norm0

                # print out convergence norms
                self.info_file.write(
                    'grad_norm0         = %e\n'%grad_norm0 +
                    '   design_norm0    = %e\n'%design_norm0 +
                    '   slack_norm0     = %e\n'%slack_norm0 +
                    'feas_norm0         = %e\n'%feas_norm0)

                # calculate convergence tolerances
                grad_tol = self.primal_tol * max(grad_norm0, 1e-3)
                feas_tol = self.cnstr_tol * max(feas_norm0, 1e-3)

            else:
                # calculate current norms
                grad_norm = dLdX._primal.norm2
                design_norm = dLdX._primal._design.norm2
                slack_norm = dLdX._primal._slack.norm2
                feas_norm = max(dLdX._dual.norm2, EPS)
                kkt_norm = np.sqrt(feas_norm**2 + grad_norm**2)

                # update the augmented Lagrangian penalty
                self.info_file.write(
                    'grad_norm          = %e (%e <-- tolerance)\n'%(
                        grad_norm, grad_tol) +
                    '   design_norm     = %e\n'%design_norm +
                    '   slack_norm      = %e\n'%slack_norm +
                    'feas_norm          = %e (%e <-- tolerance)\n\n'%(
                        feas_norm, feas_tol))

            # write convergence history
            obj_val = objective_value(X._primal._design, state)
            self._write_history(grad_norm, feas_norm, obj_val)

            # check for convergence
            if (grad_norm < grad_tol) and (feas_norm < feas_tol):
                converged = True
                break

            # compute krylov tolerances in order to achieve superlinear
            # convergence but to avoid oversolving
            krylov_tol = \
                self.tangent_KKT.krylov.rel_tol * \
                min(1.0, np.sqrt(kkt_norm/kkt_norm0))
            krylov_tol = \
                max(krylov_tol,
                    min(grad_tol/grad_norm, feas_tol/feas_norm))

            # linearize the normal and tangent matrices at this point
            self.normal_KKT.linearize(X, state)
            self.tangent_KKT.linearize(X, state, adjoint)

            # assemble the RHS vector the Lagrange multiplier solve
            normal_rhs._primal.equals(dLdX._primal)
            normal_rhs._dual.equals(0.0)
            normal_rhs.times(-1.)

            # solve for the Lagrange multiplier estimates
            self.normal_KKT.solve(normal_rhs, P, krylov_tol)

            # compute the slack term
            slack_work.exp(X._primal._slack)
            slack_work.restrict()

            # assemble the RHS vector for the normal-step solve
            # rhs = [0, 0, -(c-e^s)]
            normal_rhs.equals(0.0)
            normal_rhs._dual.equals(dLdX._dual)
            normal_rhs._dual.times(-1.)

            # solve for the normal step
            self.normal_KKT.solve(normal_rhs, normal_step, krylov_tol)

            # apply a trust radius check to the normal step
            normal_step_norm = normal_step._primal.norm2
            if normal_step_norm > 0.8*self.radius:
                normal_step._primal.times(0.8*self.radius/normal_step_norm)

            # set trust radius settings for the tangent step STCG solver
            self.tangent_KKT.radius = np.sqrt(
                self.radius**2
                - normal_step._primal._design.norm2**2
                - normal_step._primal._slack.norm2**2)

            # set up the RHS vector for the tangent-step solve
            # design component: -(W * normal_design + dL/dDesign)
            self.tangent_KKT.W.product(
                normal_step._primal._design, tangent_rhs._design)
            tangent_rhs._design.plus(dLdX._primal._design)
            tangent_rhs._design.times(-1.)
            # slack component: (Sigma * lambda)+(Sigma * Lambda * normal_slack)
            tangent_rhs._slack.equals(normal_step._primal._slack)
            tangent_rhs._slack.times(X._dual)
            tangent_rhs._slack.times(slack_work)
            dual_work.equals(X._dual)
            dual_work.times(slack_work)
            tangent_rhs._slack.plus(dual_work)
            tangent_rhs._slack.restrict()

            # solve for the tangent step
            self.tangent_KKT.solve(
                tangent_rhs, tangent_step, krylov_tol)

            # assemble the complete step
            P._primal.equals_ax_p_by(
                1., normal_step._primal, 1, tangent_step)
            P._primal._slack.restrict()

            # calculate predicted decrease in the augmented Lagrangian
            self.normal_KKT.A.product(P._primal._design, dual_work)
            slack_work.times(P._primal._slack)
            slack_work.times(-1.)
            dual_work.plus(slack_work)
            dual_work.plus(dLdX._dual)
            pred = self.tangent_KKT.pred \
                - normal_step._primal.inner(dLdX._primal) \
                + self.mu*dLdX._dual.norm2**2 \
                - P._dual.inner(dual_work) \
                - self.mu*dual_work.inner(dual_work)

            # apply globalization
            if self.globalization == 'trust':
                old_flag = min_radius_active
                success, min_radius_active = self.trust_step(
                    X, state, P, adjoint, dLdX, pred, krylov_tol, feas_tol,
                    state_work, dual_work, slack_work, kkt_work,
                    normal_step, normal_rhs, tangent_step, tangent_rhs)

                # watchdog on trust region failures
                if min_radius_active and old_flag:
                    self.info_file.write(
                        'Trust radius breakdown! Terminating...\n')
                    break
            elif self.globalization is None:
                # accept step
                X._primal.plus(P._primal)
                X._primal._slack.restrict()
                X._dual.plus(P._dual)

                # calculate states
                state.equals_primal_solution(X._primal._design)

                # if this is a matrix-based problem, tell the solver to factor
                # some important matrices to be used in the next iteration
                if self.factor_matrices and self.iter < self.max_iter:
                    factor_linear_system(X._primal._design, state)

                # perform an adjoint solution for the Lagrangian
                state_work.equals_objective_partial(X._primal._design, state)
                dCdU(X._primal._design, state).T.product(X._dual, adjoint)
                state_work.plus(adjoint)
                state_work.times(-1.)
                dRdU(X._primal._design, state).T.solve(state_work, adjoint)

            # send current solution info to the user
            solver_info = current_solution(
                X._primal._design, state, adjoint, X._dual, self.iter)
            if isinstance(solver_info, str):
                self.info_file.write('\n' + solver_info + '\n')

        ############################
        # END OF NEWTON LOOP

        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Optimization FAILED!\n')

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n\n'%self.iter)

    def trust_step(self, X, state, P, adjoint, dLdX, pred, krylov_tol, feas_tol,
                   state_work, dual_work, slack_work, step_save,
                   normal_step, normal_rhs, tangent_step, tangent_rhs):
        # start trust region loop
        max_iter = 5
        iters = 0
        min_radius_active = False
        converged = False
        self.info_file.write('\n')
        while iters <= max_iter:
            iters += 1
            # evaluate the constraint term at the current step
            dual_work.equals_constraints(X._primal._design, state)
            slack_work.exp(X._primal._slack)
            slack_work.times(-1.)
            slack_work.restrict()
            dual_work.plus(slack_work)

            # compute the merit value at the current step
            merit_init = objective_value(X._primal._design, state) \
                + X._dual.inner(dual_work) \
                + 0.5*self.mu*(dual_work.norm2**2)

            # add the FLECS step
            X.plus(P)

            # solve states at the new step
            state_work.equals_primal_solution(X._primal._design)

            # evaluate the constraint terms at the new step
            dual_work.equals_constraints(X._primal._design, state)
            slack_work.exp(X._primal._slack)
            slack_work.times(-1.)
            slack_work.restrict()
            dual_work.plus(slack_work)

            # compute the merit value at the new step
            merit_next = objective_value(X._primal._design, state) \
                + X._dual.inner(dual_work) \
                + 0.5*self.mu*(dual_work.norm2**2)

            # reset the step
            X.minus(P)

            # evaluate the quality of the FLECS model
            rho = (merit_init - merit_next)/pred

            self.info_file.write(
                'Trust Region Step : iter %i\n'%iters +
                '   primal_step    = %e\n'%P._primal.norm2 +
                '      design_step = %e\n'%P._primal._design.norm2 +
                '      slack_step  = %e\n'%P._primal._slack.norm2 +
                '   lambda_step    = %e\n'%P._dual.norm2 +
                '\n' +
                '   merit_init     = %e\n'%merit_init +
                '   merit_next     = %e\n'%merit_next +
                '   pred           = %e\n'%pred +
                '   rho            = %f\n'%rho)

            # modify radius based on model quality
            if rho <= 0.01 or np.isnan(rho):
                # model is bad! -- first we try a 2nd order correction
                if iters == 1:
                    # save the old step in case correction fails
                    step_save.equals(P)

                    # recalculate the normal-step RHS
                    normal_rhs.equals(0.0)
                    normal_rhs._dual.equals(dual_work)
                    normal_rhs._dual.times(-1.)

                    # attempt a 2nd order correction
                    self.info_file.write(
                        '   Attempting a second order correction...\n')
                    self.normal_KKT.solve(normal_rhs, P, krylov_tol)

                elif iters == 2:
                    # if we got here, the second order correction failed
                    # reject step
                    self.info_file.write(
                        '   Correction failed! Resetting step...\n')
                    P.equals(step_save)

                else:
                    self.radius = max(0.5*self.radius, self.min_radius)
                    if self.radius == self.min_radius:
                        self.info_file.write(
                            '      Reached minimum radius! ' +
                            'Exiting globalization...\n')
                        min_radius_active = True
                        break

                    else:
                        self.info_file.write(
                            '   Re-solving with smaller radius -> ' +
                            '%f\n'%self.radius)

                        # shrink the normal step
                        normal_step._primal.times(
                            0.8*self.radius/normal_step._primal.norm2)

                        # calculate the tangent step Radius
                        self.tangent_KKT.radius = np.sqrt(
                            self.radius**2
                            - normal_step._primal._design.norm2**2
                            - normal_step._primal._slack.norm2**2)

                        # set up the RHS vector for the tangent-step solve
                        self.tangent_KKT.W.product(
                            normal_step._primal._design, tangent_rhs._design)
                        tangent_rhs._design.plus(dLdX._primal._design)
                        tangent_rhs._design.times(-1.)
                        tangent_rhs._slack.equals(normal_step._primal._slack)
                        tangent_rhs._slack.times(X._dual)
                        tangent_rhs._slack.times(slack_work)
                        dual_work.equals(X._dual)
                        dual_work.times(slack_work)
                        tangent_rhs._slack.plus(dual_work)
                        tangent_rhs._slack.restrict()

                        # solve for the tangent step
                        self.tangent_KKT.solve(
                            tangent_rhs, tangent_step, krylov_tol)

                        # assemble the complete step
                        P._primal.equals_ax_p_by(
                            1., normal_step._primal, 1, tangent_step)
                        P._primal._slack.restrict()

                        # calculate predicted decrease in the merit function
                        self.normal_KKT.A.product(P._primal._design, dual_work)
                        slack_work.exp(X._primal._slack)
                        slack_work.restrict()
                        slack_work.times(P._primal._slack)
                        slack_work.times(-1.)
                        dual_work.plus(slack_work)
                        dual_work.plus(dLdX._dual)
                        pred = self.tangent_KKT.pred \
                            - normal_step._primal.inner(dLdX._primal) \
                            + self.mu*dLdX._dual.norm2**2 \
                            - P._dual.inner(dual_work) \
                            - self.mu*dual_work.inner(dual_work)
            else:
                if iters == 2:
                    # 2nd order correction worked -- yay!
                    self.info_file.write('   Correction worked!\n')

                # model is okay -- accept primal step
                self.info_file.write('\nStep accepted!\n')

                # accept the new step entirely
                X.plus(P)
                X._primal._slack.restrict()
                state.equals_primal_solution(X._primal._design)

                # if this is a matrix-based problem, tell the solver to factor
                # some important matrices to be used in the next iteration
                if self.factor_matrices and self.iter < self.max_iter:
                    factor_linear_system(X._primal._design, state)

                # evaluate constraints
                dual_work.equals_constraints(X._primal._design, state)
                slack_work.exp(X._primal._slack)
                slack_work.times(-1.)
                dual_work.plus(slack_work)

                # update the penalty coefficient
                feas_norm = dual_work.norm2
                if feas_norm > feas_tol:
                    if feas_norm <= self.eta:
                        # constraints are good
                        # tighten tolerances
                        self.eta = self.eta/(self.mu**0.9)
                    else:
                        # constraints are bad
                        # increase penalty if we haven't met feasibility
                        self.mu = min(self.mu*(10.**self.mu_pow), self.mu_max)
                        self.eta = 1./(self.mu**0.1)
                        self.info_file.write(
                            '   Mu increased -> %1.2e\n'%self.mu)

                # perform an adjoint solution for the Lagrangian
                state_work.equals_objective_partial(X._primal._design, state)
                dCdU(X._primal._design, state).T.product(X._dual, adjoint)
                state_work.plus(adjoint)
                state_work.times(-1.)
                dRdU(X._primal._design, state).T.solve(state_work, adjoint)

                # check the trust radius
                if self.tangent_KKT.trust_active:
                    # if active, decide if we want to increase it
                    self.info_file.write('Trust radius active...\n')
                    if rho >= 0.5:
                        # model is good enough -- increase radius
                        self.radius = min(2.*self.radius, self.max_radius)
                        self.info_file.write(
                            '   Radius increased -> %f\n'%self.radius)
                        min_radius_active = False

                # trust radius globalization worked, break loop
                converged = True
                self.info_file.write('\n')
                break

        return converged, min_radius_active
