from kona.algorithms.base_algorithm import OptimizationAlgorithm

class CompositeStepRSNK(OptimizationAlgorithm):
    """
    A reduced-space composite-step optimization algorithm for PDE-governed
    (in)equality constrained problems.

    This algorithm uses a novel 2nd order adjoint formulation for constraint
    jacobian and constrained hessian products.

    Parameters
    ----------
    primal_factory : VectorFactory
    state_factory : VectorFactory
    eq_factory : VectorFactory
    ineq_factory : VectorFactory
    optns : dict, optional
    """
    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(CompositeStepRSNK, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns)

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(9)
        self.state_factory.request_num_vectors(4)
        self.eq_factory.request_num_vectors(16)

        # get general options
        self.factor_matrices = get_opt(self.optns, False, 'matrix_explicit')

        # get trust region options
        self.radius = get_opt(self.optns, 0.5, 'trust', 'init_radius')
        self.min_radius = get_opt(self.optns, 0.5/(2**3), 'trust', 'min_radius')
        self.max_radius = get_opt(self.optns, 0.5*(2**3), 'trust', 'max_radius')

        # get penalty parameter options
        self.mu = get_opt(self.optns, 1.0, 'penalty', 'mu_init')
        self.mu_pow = get_opt(self.optns, 1e-8, 'penalty', 'mu_pow')
        self.mu_max = get_opt(self.optns, 1e4, 'penalty', 'mu_max')

        # get globalization type
        self.globalization = get_opt(self.optns, 'linesearch', 'globalization')

        if self.globalization not in ['trust', 'linesearch', None]:
            raise TypeError(
                'Invalid globalization type! ' +
                'Can only use \'trust\' or \'linesearch\'. ' +
                'If you want to skip globalization, set to None.')

        # initialize the KKT matrix definition
        normal_optns = get_opt(self.optns, {}, 'composite-step', 'normal-step')
        self.normal_KKT = AugmentedKKTMatrix(
            [self.primal_factory, self.state_factory, self.eq_factory],
            normal_optns)
        tangent_optns = get_opt(self.optns, {}, 'composite-step', 'tangent-step')
        self.tangent_KKT = LagrangianHessian(
            [self.primal_factory, self.state_factory, self.eq_factory],
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
            '%7i'%self.iter + ' '*5 +
            '%7i'%self.primal_factory._memory.cost + ' '*5 +
            '%11e'%opt + ' '*5 +
            '%11e'%feas + ' '*5 +
            '%11e'%obj + ' '*5 +
            '%11e'%self.mu + ' '*5 +
            '%11e'%self.radius + '\n'
        )

    def _generate_primal(self):
        return self.primal_factory.generate()

    def _generate_dual(self):
        return self.eq_factory.generate()

    def _generate_KKT_vector(self):
        primal = self._generate_primal()
        dual = self._generate_dual()
        return ReducedKKTVector(primal, dual)

    def _generate_all_memory(self):
        # generate composite KKT vectors
        self.X = self._generate_KKT_vector()
        self.P = self._generate_KKT_vector()
        self.dLdX = self._generate_KKT_vector()
        self.kkt_work = self._generate_KKT_vector()
        self.normal_rhs = self._generate_KKT_vector()
        self.normal_step = self._generate_KKT_vector()
        self.tangent_rhs = self._generate_primal()
        self.tangent_step = self._generate_primal()

        # generate primal vectors
        self.design_work = self.primal_factory.generate()
        self.design_save = self.primal_factory.generate()

        # generate state vectors
        self.state = self.state_factory.generate()
        self.state_work = self.state_factory.generate()
        self.adjoint = self.state_factory.generate()
        self.adjoint_work = self.state_factory.generate()

        # generate dual vectors
        self.dual_work = self.eq_factory.generate()
        self.slack_work = self.eq_factory.generate()

    def solve(self):
        self._write_header()
        self.info_file.write(
            '\n' +
            '**************************************************\n' +
            '***       Using Composite-Step Algorithm       ***\n' +
            '**************************************************\n' +
            '\n')

        # generate all the vectors used in the optimization
        self._generate_all_memory()

        # do some aliasing
        X = self.X
        state = self.state
        adjoint = self.adjoint
        dLdX = self.dLdX

        P = self.P
        normal_rhs = self.normal_rhs
        normal_step = self.normal_step
        tangent_rhs = self.tangent_rhs
        tangent_step = self.tangent_step

        design_work = self.design_work
        dual_work = self.dual_work
        slack_work = self.slack_work
        state_work = self.state_work

        # initialize basic data for outer iterations
        converged = False
        self.iter = 0

        # evaluate the initial design before starting outer iterations
        X.equals_init_guess()
        if not state.equals_primal_solution(X.primal):
            raise RuntimeError(
                'Invalid initial guess! Nonlinear solution breakdown.')

        if self.factor_matrices and self.iter < self.max_iter:
            factor_linear_system(X.primal, state)

        # perform an adjoint solution for the Lagrangian
        state_work.equals_objective_partial(X.primal, state)
        dCdU(X.primal, state).T.product(X.dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X.primal, state).T.solve(state_work, adjoint)

        # send initial point info to the user
        solver_info = current_solution(
            self.iter, X.primal, state, adjoint, X.dual)
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
            dLdX.equals_KKT_conditions(X, state, adjoint)
            # print info on current point
            self.info_file.write(
                '==========================================================\n' +
                'Beginning Major Iteration %i\n\n'%self.iter)
            self.info_file.write(
                'primal vars        = %e\n'%X.primal.norm2)
            self.info_file.write(
                'multipliers        = %e\n\n'%X.dual.norm2)

            if self.iter == 1:
                # calculate initial norms
                grad_norm0 = dLdX.primal.norm2
                feas_norm0 = max(dLdX.dual.norm2, EPS)
                kkt_norm0 = np.sqrt(feas_norm0**2 + grad_norm0**2)

                # set current norms to initial
                kkt_norm = kkt_norm0
                grad_norm = grad_norm0
                feas_norm = feas_norm0

                # print out convergence norms
                self.info_file.write(
                    'grad_norm0         = %e\n'%grad_norm0 +
                    'feas_norm0         = %e\n'%feas_norm0)

                # calculate convergence tolerances
                grad_tol = self.primal_tol * max(grad_norm0, 1e-3)
                feas_tol = self.cnstr_tol * max(feas_norm0, 1e-3)

            else:
                # calculate current norms
                grad_norm = dLdX.primal.norm2
                feas_norm = max(dLdX.dual.norm2, EPS)
                kkt_norm = np.sqrt(feas_norm**2 + grad_norm**2)

                # update the augmented Lagrangian penalty
                self.info_file.write(
                    'grad_norm          = %e (%e <-- tolerance)\n'%(
                        grad_norm, grad_tol) +
                    'feas_norm          = %e (%e <-- tolerance)\n'%(
                        feas_norm, feas_tol))

            # write convergence history
            obj_val = objective_value(X.primal, state)
            self._write_history(grad_norm, feas_norm, obj_val)

            # check for convergence
            if (grad_norm < grad_tol) and (feas_norm < feas_tol):
                converged = True
                break

            # compute krylov tolerances in order to achieve superlinear
            # convergence but to avoid oversolving
            self.krylov_tol = \
                self.tangent_KKT.krylov.rel_tol * \
                min(1.0, np.sqrt(kkt_norm/kkt_norm0))
            self.krylov_tol = \
                max(krylov_tol,
                    min(grad_tol/grad_norm, feas_tol/feas_norm))

            # linearize the normal and tangent matrices at this point
            self.normal_KKT.linearize(X, state)
            self.tangent_KKT.linearize(X, state, adjoint)

            # assemble the RHS vector the Lagrange multiplier solve
            normal_rhs.primal.equals(dLdX.primal)
            normal_rhs.dual.equals(0.0)
            normal_rhs.times(-1.)

            # solve for the Lagrange multiplier estimates
            self.normal_KKT.solve(normal_rhs, P, self.krylov_tol)

            # assemble the RHS vector for the normal-step solve
            normal_rhs.equals(0.0)
            normal_rhs.dual.equals(dLdX.dual)
            normal_rhs.dual.times(-1.)

            # solve for the normal step
            self.normal_KKT.solve(normal_rhs, normal_step, krylov_tol)

            # apply a trust radius check to the normal step
            normal_step_norm = normal_step.primal.norm2
            if normal_step_norm > 0.8*self.radius:
                normal_step.primal.times(0.8 * self.radius / normal_step_norm)

            # set trust radius settings for the tangent step STCG solver
            self.tangent_KKT.radius = np.sqrt(
                self.radius**2 +
                normal_step.primal.norm2 ** 2)

            # set up the RHS vector for the tangent-step solve
            # design component: -(W * normal_design + dL/dDesign)
            self.tangent_KKT.multiply_W(
                normal_step.primal, tangent_rhs)
            tangent_rhs.plus(dLdX.primal)
            tangent_rhs.times(-1.)

            # solve for the tangent step
            self.tangent_KKT.solve(
                tangent_rhs, tangent_step, krylov_tol)

            # assemble the complete step
            P.primal.equals_ax_p_by(
                1., normal_step.primal, 1, tangent_step)

            # calculate predicted decrease in the merit function
            self.calc_pred_reduction()

            # apply globalization
            if self.globalization == 'trust':
                old_flag = min_radius_active
                success, min_radius_active = self.trust_step()

                # watchdog on trust region failures
                if min_radius_active and old_flag:
                    self.info_file.write(
                        'Trust radius breakdown! Terminating...\n')
                    break

            elif self.globalization == 'linesearch':
                old_flag = min_radius_active
                success, min_radius_active = self.backtracking_step()

                # watchdog on trust region failures
                if min_radius_active and old_flag:
                    self.info_file.write(
                        'Trust radius breakdown! Terminating...\n')
                    break

            elif self.globalization is None:
                # add the full step
                X.primal.plus(P.primal)
                X.dual.plus(P.dual)

                # solve states at the new step
                state.equals_primal_solution(X.primal)

                # if this is a matrix-based problem, tell the solver to factor
                # some important matrices to be used in the next iteration
                if self.factor_matrices and self.iter < self.max_iter:
                    factor_linear_system(X.primal, state)

                # perform an adjoint solution for the Lagrangian
                state_work.equals_objective_partial(X.primal, state)
                dCdU(X.primal, state).T.product(X.dual, adjoint)
                state_work.plus(adjoint)
                state_work.times(-1.)
                dRdU(X.primal, state).T.solve(state_work, adjoint)

            # send current solution info to the user
            solver_info = current_solution(
                self.iter, X.primal, state, adjoint, X.dual)
            if isinstance(solver_info, str):
                self.info_file.write('\n' + solver_info + '\n')

        ############################
        # END OF NEWTON LOOP

        self.info_file.write('\n')
        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Optimization FAILED!\n')

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n\n'%self.iter)

    def eval_merit(self, primal, state, dual, cnstr):
        return objective_value(primal, state) \
            + dual.inner(cnstr) \
            + 0.5*self.mu*(cnstr.norm2**2)

    def calc_pred_reduction(self):
        # calculate predicted decrease in the augmented Lagrangian
        self.normal_KKT.A.product(self.P.primal, self.dual_work)
        self.dual_work.plus(self.dLdX.dual)
        self.tangent_rhs.minus(self.dLdX.primal)
        self.tangent_rhs.times(0.5)
        self.pred_reduction = self.tangent_KKT.pred \
            + self.normal_step.primal.inner(self.tangent_rhs) \
            + 0.5*self.mu* self.dLdX.dual.norm2 ** 2 \
            - self.P.dual.inner(self.dual_work) \
            - 0.5*self.mu*self.dual_work.norm2**2

        # calculate the new penalty parameter if necessary
        denom = 0.25*(self.dLdX.dual.norm2 ** 2 - self.dual_work.norm2 ** 2)
        if self.pred_reduction < self.mu*denom:
            self.mu += -self.pred_reduction/denom + self.mu_pow
            self.info_file.write('\n')
            self.info_file.write('   Mu updated -> %e\n'%self.mu)
            # recalculate the prediction with new penalty
            self.pred_reduction = self.tangent_KKT.pred \
                + self.normal_step.primal.inner(self.tangent_rhs) \
                + 0.5*self.mu* self.dLdX.dual.norm2 ** 2 \
                - self.P.dual.inner(self.dual_work) \
                - 0.5*self.mu*self.dual_work.norm2**2

    def backtracking_step(self):
        # do some aliasing
        X = self.X
        state = self.state
        P = self.P
        adjoint = self.adjoint
        dLdX = self.dLdX
        design_work = [self.design_work, self.design_save]
        state_work = self.state_work
        adjoint_work = self.adjoint_work
        dual_work = self.dual_work
        slack_work = self.slack_work
        kkt_work = self.kkt_work

        # compute the merit value at the current step
        f_init = self.eval_merit(
            X.primal, state, X.dual, dLdX.dual)

        # compute the merit function derivative w.r.t. design
        design_work[0].equals(dLdX.primal)
        dCdX(X.primal, state).T.product(dLdX.dual, design_work[1])
        design_work[1].times(self.mu)
        design_work[0].plus(design_work[1])
        dCdU(X.primal, state).T.product(dLdX.dual, state_work)
        state_work.times(-1.)
        dRdU(X.primal, state).T.solve(state_work, adjoint_work)
        dRdX(X.primal, state).T.product(adjoint_work, design_work[1])
        design_work[1].times(self.mu)
        design_work[0].plus(design_work[1])

        # compute the directional derivative for the merit function
        p_dot_grad = design_work[0].inner(P.primal)

        self.info_file.write('\n')
        self.info_file.write(
            '   primal_step    = %e\n' % P.primal.norm2 +
            '   lambda_step    = %e\n' % P.dual.norm2 +
            '\n' +
            '   p_dot_grad     = %e\n' % p_dot_grad)

        # if p_dot_grad >= 0:
        #     raise ValueError('Search direction is not a descent direction!')

        # start line search iterations
        max_iter = 5
        iters = 0
        min_radius_active = False
        converged = False
        decr_cond = 1e-4
        rdtn_factor = 0.5
        min_alpha = 1e-3
        alpha = 1.0
        self.info_file.write('\n')
        while iters <= max_iter:
            iters += 1

            self.info_file.write(
                'Back-tracking : iter %i\n'%iters +
                '   alpha      = %f\n'%alpha)

            # calculate the next step
            kkt_work.primal.equals_ax_p_by(1., X.primal, alpha, P.primal)
            kkt_work.dual.equals_ax_p_by(1., X.dual, 1., P.dual)

            # solve for the states
            if state_work.equals_primal_solution(kkt_work.primal):
                # evaluate constraints
                dual_work.equals_constraints(kkt_work.primal, state_work)

                # calculate the merit function
                f_next = self.eval_merit(
                    kkt_work.primal, state_work,
                    kkt_work.dual, dual_work)

                # calculate sufficient decrease
                f_suff = f_init + decr_cond*alpha*p_dot_grad

            else:
                # state solution failed!
                f_suff = np.nan
                f_next = np.nan

            # evaluate step
            self.info_file.write(
                '   f_suff     = %e\n'%f_suff +
                '   f_next     = %e\n'%f_next)
            if f_next <= f_suff:
                self.info_file.write(
                    'Line search succeeded!\n')
                converged = True
                break
            else:
                self.info_file.write(
                    '   Bad step! Back-tracking...\n')
                alpha = max(rdtn_factor*alpha, min_alpha)
                if alpha == min_alpha:
                    self.info_file.write(
                        'Minimum step reached! Terminating...\n')
                    break

        # deal with step acceptance
        shrink = False
        if converged:
            # line search converged, so we accept step
            X.equals(kkt_work)
            state.equals(state_work)

            # if this is a matrix-based problem, tell the solver to
            # factor some important matrices to be used in the next
            # iteration
            if self.factor_matrices and self.iter < self.max_iter:
                factor_linear_system(X.primal, state)

            # perform an adjoint solution for the Lagrangian
            state_work.equals_objective_partial(X.primal, state)
            dCdU(X.primal, state).T.product(X.dual, adjoint)
            state_work.plus(adjoint)
            state_work.times(-1.)
            dRdU(X.primal, state).T.solve(state_work, adjoint)

            # flag for trust radius shrinking
            if alpha < 1:
                shrink = True
        else:
            # line search failed, need to shrink trust radius
            shrink = True

        if shrink:
            # shrink trust radius
            if self.radius > self.min_radius:
                self.radius = \
                    max(alpha * P.primal.norm2, self.min_radius)
                self.info_file.write(
                    '   Radius shrunk -> %f\n'%self.radius)
            else:
                self.info_file.write(
                    '   Reached minimum radius!\n')
                min_radius_active = True
        else:
            # increase trust radius if it was active
            if self.tangent_KKT.trust_active:
                self.info_file.write('   Trust radius active...\n')
                if self.radius < self.max_radius:
                    self.radius = \
                        min(2.*self.radius, self.max_radius)
                    self.info_file.write(
                        '   Radius increased -> %f\n'%self.radius)
                else:
                    self.info_file.write(
                        '   Max radius reached!\n')
                min_radius_active = False

        self.info_file.write('\n')
        return converged, min_radius_active

    def trust_step(self):
        # do some aliasing
        X = self.X
        P = self.P
        state = self.state
        adjoint = self.adjoint
        dLdX = self.dLdX
        dual_work = self.dual_work
        slack_work = self.slack_work
        state_work = self.state_work
        kkt_work = self.kkt_work
        tangent_rhs = self.tangent_rhs
        tangent_step = self.tangent_step
        normal_step = self.normal_step

        # compute the merit value at the current step
        merit_init = self.eval_merit(
            X.primal, state, X.dual, dLdX.dual)

        # start trust region loop
        max_iter = 5
        iters = 0
        min_radius_active = False
        converged = False
        self.info_file.write('\n')
        while iters <= max_iter:
            iters += 1
            # save current point
            kkt_work.equals(X)

            # get the new design point
            X.primal.plus(P.primal)
            X.dual.plus(P.dual)

            # solve states at the new step
            if state_work.equals_primal_solution(X.primal):
                # evaluate the constraint terms at the new step
                dual_work.equals_constraints(X.primal, state_work)

                # compute the merit value at the new step
                merit_next = self.eval_merit(
                    X.primal, state_work, X.dual, dual_work)

                # evaluate the quality of the FLECS model
                rho = (merit_init - merit_next)/self.pred_reduction
            else:
                merit_next = np.nan
                rho = np.nan

            # reset step back
            X.equals(kkt_work)

            self.info_file.write(
                'Trust Region Step : iter %i\n' % iters +
                '   primal_step    = %e\n' % P.primal.norm2 +
                '   lambda_step    = %e\n' % P.dual.norm2 +
                '\n' +
                '   merit_init     = %e\n' % merit_init +
                '   merit_next     = %e\n' % merit_next +
                '   pred           = %e\n' % self.pred_reduction +
                '   rho            = %e\n' % rho)

            # modify radius based on model quality
            if rho <= 0.01 or np.isnan(rho):
                # model is bad! -- shrink radius and re-solve
                self.radius = max(0.5 * P.primal.norm2, self.min_radius)
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

                    # apply a trust radius check to the normal step
                    normal_step_norm = normal_step.primal.norm2
                    if normal_step_norm > 0.8*self.radius:
                        normal_step.primal.times(
                            0.8*self.radius/normal_step_norm)

                    # calculate the tangent step radius
                    self.tangent_KKT.radius = np.sqrt(
                        self.radius ** 2
                        - normal_step.primal.norm2 ** 2)

                    # set up the RHS vector for the tangent-step solve
                    self.tangent_KKT.multiply_W(
                        normal_step.primal, tangent_rhs)
                    tangent_rhs.plus(dLdX.primal)
                    tangent_rhs.times(-1.)

                    # solve for the tangent step
                    self.tangent_KKT.solve(
                        tangent_rhs, tangent_step, self.krylov_tol)

                    # assemble the complete step
                    P.primal.equals_ax_p_by(
                        1., normal_step.primal, 1, tangent_step)
                    P.primal.slack.restrict()

                    # calculate predicted decrease in the augmented Lagrangian
                    self.calc_pred_reduction()
            else:
                # model is okay -- accept primal step
                self.info_file.write('\nStep accepted!\n')
                X.primal.plus(P.primal)
                X.dual.plus(P.dual)

                # solve states at the new step
                state.equals_primal_solution(X.primal)

                # if this is a matrix-based problem, tell the solver to factor
                # some important matrices to be used in the next iteration
                if self.factor_matrices and self.iter < self.max_iter:
                    factor_linear_system(X.primal, state)

                # perform an adjoint solution for the Lagrangian
                state_work.equals_objective_partial(X.primal, state)
                dCdU(X.primal, state).T.product(X.dual, adjoint)
                state_work.plus(adjoint)
                state_work.times(-1.)
                dRdU(X.primal, state).T.solve(state_work, adjoint)

                # check the trust radius
                if self.tangent_KKT.trust_active:
                    # if active, decide if we want to increase it
                    self.info_file.write('Trust radius active...\n')
                    if rho >= 0.5:
                        # model is good enough -- increase radius
                        if self.radius < self.max_radius:
                            self.radius = min(2.*self.radius, self.max_radius)
                            self.info_file.write(
                                '   Radius increased -> %f\n'%self.radius)
                        else:
                            self.info_file.write(
                                '   Max radius reached!\n')
                        min_radius_active = False

                # trust radius globalization worked, break loop
                converged = True
                self.info_file.write('\n')
                break

        return converged, min_radius_active

# imports here to prevent circular errors
import numpy as np
from kona.options import get_opt
from kona.linalg.common import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import dCdX, dCdU, dRdX, dRdU
from kona.linalg.matrices.hessian import AugmentedKKTMatrix, LagrangianHessian
from kona.linalg.solvers.util import EPS