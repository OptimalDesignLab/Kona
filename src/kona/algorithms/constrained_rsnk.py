from kona.algorithms.base_algorithm import OptimizationAlgorithm


class ConstrainedRSNK(OptimizationAlgorithm):
    """
    A reduced-space Newton-Krylov optimization algorithm for PDE-governed
    equality constrained problems, globalized with a trust-region approach.

    This algorithm uses a 2nd order adjoint formulation of the KKT matrix-vector 
    product, in conjunction with a novel Krylov-method called 
    `FLECS<http://dx.doi.org/10.1137/140994496>`_ for non-convex saddle point problems.

    More information on this reduced-space Newton-Krylov appoach can be found 
    `in this paper <http://dx.doi.org/10.1007/s00158-017-1734-0>`_.

    Attributes
    ----------
    grad_norm0, feas_norm0, kkt_norm0 : float
        Initial optimality norms.
    iter : int
        Optimization iteration counter.
    factor_matrices : bool
        Boolean flag for matrix-based PDE solvers.
    radius, min_radius, max_radius : float
        Trust radius parameters.
    mu, mu_init, mu_max, mu_pow, eta : float
        Augmented Lagrangian constraint factor parameters.
    scale, grad_scale, feas_scale : float
        Optimality metric normalization factors.
    KKT_matrix : :class:`~kona.linalg.matrices.hessian.ReducedKKTVector`
        Matrix object defining the KKT matrix-vector product.
    precond : :class:`~kona.linalg.matrices.hessian.basic.BaseHessian`-like
        Matrix object defining the preconditioner to the KKT system.
    krylov : :class:`~kona.linalg.solvers.krylov.FLECS`
        A krylov solver object used to solve the system defined by this matrix.
    globalization : string
        Flag to determine solution globalization type.
    """

    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(ConstrainedRSNK, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(6 + 1)
        self.state_factory.request_num_vectors(3)
        self.eq_factory.request_num_vectors(12 + 2)

        # misc variables
        self.grad_norm0 = EPS
        self.feas_norm0 = EPS
        self.kkt_norm0 = EPS
        self.iter = 0

        # general options
        ############################################################
        self.factor_matrices = get_opt(self.optns, False, 'matrix_explicit')

        # trust radius settings
        ############################################################
        self.radius = get_opt(self.optns, 0.5, 'trust', 'init_radius')
        self.min_radius = get_opt(self.optns, 0.5/(2**3), 'trust', 'min_radius')
        self.max_radius = get_opt(self.optns, 0.5*(2**3), 'trust', 'max_radius')

        # augmented Lagrangian settings
        ############################################################
        self.mu = get_opt(self.optns, 1.0, 'penalty', 'mu_init')
        self.mu_init = self.mu
        self.mu_pow = get_opt(self.optns, 0.5, 'penalty', 'mu_pow')
        self.mu_max = get_opt(self.optns, 1e5, 'penalty', 'mu_max')
        self.eta = 1./(self.mu**0.1)

        # reduced KKT settings
        ############################################################
        self.nu = get_opt(self.optns, 0.95, 'rsnk', 'nu')
        reduced_optns = get_opt(self.optns, {}, 'rsnk')
        reduced_optns['out_file'] = self.info_file
        self.KKT_matrix = ReducedKKTMatrix(
            [self.primal_factory, self.state_factory,
             self.eq_factory],
            reduced_optns)
        self.mat_vec = self.KKT_matrix.product

        # KKT system preconditiner settings
        ############################################################
        self.precond = get_opt(self.optns, None, 'rsnk', 'precond')
        self.idf_schur = None
        if self.precond is None:
            # use identity matrix product as preconditioner
            self.eye = IdentityMatrix()
            self.precond = self.eye.product
        elif self.precond is 'idf_schur':
            self.idf_schur = ReducedSchurPreconditioner(
                [primal_factory, state_factory, eq_factory, ineq_factory])
            self.precond = self.idf_schur.product
        else:
            raise BadKonaOption(self.optns, 'rsnk', 'precond')

        # krylov solver settings
        ############################################################
        krylov_optns = {
            'krylov_file':get_opt(
                self.optns, 'kona_krylov.dat', 'rsnk', 'krylov_file'),
            'subspace_size':get_opt(self.optns, 10, 'rsnk', 'subspace_size'),
            'check_res':get_opt(self.optns, True, 'rsnk', 'check_res'),
            'rel_tol':get_opt(self.optns, 1e-2, 'rsnk', 'rel_tol'),
        }
        self.krylov = FLECS(
            [self.primal_factory, self.eq_factory],
            krylov_optns)

        # get globalization options
        ############################################################
        self.globalization = get_opt(self.optns, 'trust', 'globalization')
        if self.globalization not in ['trust', 'filter', None]:
            raise TypeError(
                'Invalid globalization! ' +
                'Can only use \'trust\' or \'filter\'. ' +
                'If you want to skip globalization, set to None.')
        else:
            if self.globalization == 'filter':
                self.filter = SimpleFilter()

    def _write_header(self):
        self.hist_file.write(
            '# Kona constrained RSNK convergence history file\n' +
            '# iters' + ' '*5 +
            '   cost' + ' '*5 +
            'optimality  ' + ' '*5 +
            'feasibility ' + ' '*5 +
            'objective   ' + ' '*5 +
            'mu param    ' + ' '*5 +
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

    def _generate_KKT_vector(self):
        primal = self.primal_factory.generate()
        dual = self.eq_factory.generate()
        return ReducedKKTVector(primal, dual)

    def solve(self):
        self._write_header()
        self.info_file.write(
            '\n' +
            '**************************************************\n' +
            '***        Using FLECS-based Algorithm         ***\n' +
            '**************************************************\n' +
            '\n')

        # generate composite KKT vectors
        X = self._generate_KKT_vector()
        P = self._generate_KKT_vector()
        dLdX = self._generate_KKT_vector()
        kkt_rhs = self._generate_KKT_vector()
        kkt_save = self._generate_KKT_vector()
        kkt_work = self._generate_KKT_vector()

        # generate state vectors
        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        adjoint = self.state_factory.generate()

        # generate dual vectors
        dual_work = self.eq_factory.generate()

        # initialize basic data for outer iterations
        converged = False

        # evaluate the initial design before starting outer iterations
        X.equals_init_guess()
        state.equals_primal_solution(X.primal)
        if self.factor_matrices and self.iter < self.max_iter:
            factor_linear_system(X.primal, state)

        # perform an adjoint solution for the Lagrangian
        adjoint.equals_lagrangian_adjoint(X, state, state_work)

        # send initial point info to the user
        solver_info = current_solution(self.iter, X.primal, state, adjoint,
                                       X.dual)
        if isinstance(solver_info, str):
            self.info_file.write('\n' + solver_info + '\n')

        # BEGIN NEWTON LOOP HERE
        ###############################
        min_radius_active = False
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
                self.grad_norm0 = dLdX.primal.norm2
                self.feas_norm0 = max(dLdX.dual.norm2, EPS)
                self.kkt_norm0 = np.sqrt(
                    self.feas_norm0**2 + self.grad_norm0**2)

                # set current norms to initial
                kkt_norm = self.kkt_norm0
                grad_norm = self.grad_norm0
                feas_norm = self.feas_norm0

                # print out convergence norms
                self.info_file.write(
                    'grad_norm0         = %e\n'%self.grad_norm0 +
                    'feas_norm0         = %e\n'%self.feas_norm0)

                # calculate convergence tolerances
                grad_tol = self.primal_tol
                feas_tol = self.cnstr_tol

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

            # update penalty term
            ref_norm = min(grad_norm, feas_norm)
            self.mu = max(
                self.mu,
                self.mu_init*((self.feas_norm0/ref_norm)**self.mu_pow))
            self.mu = min(self.mu, self.mu_max)

            # write convergence history
            obj_val = objective_value(X.primal, state)
            self._write_history(grad_norm, feas_norm, obj_val)

            # check for convergence
            if (grad_norm < grad_tol) and (feas_norm < feas_tol):
                converged = True
                break

            # compute krylov tolerances in order to achieve superlinear
            # convergence but to avoid oversolving
            krylov_tol = self.krylov.rel_tol*min(
                1.0, np.sqrt(kkt_norm/self.kkt_norm0))
            krylov_tol = max(krylov_tol,
                             min(grad_tol/grad_norm,
                                 feas_tol/feas_norm))
            self.info_file.write('\nkrylov tol = %e\n'%krylov_tol)

            # set other solver and product options
            self.krylov.rel_tol = krylov_tol
            self.krylov.radius = self.radius
            self.krylov.mu = self.mu

            # linearize the KKT matrix
            self.KKT_matrix.linearize(X, state, adjoint)
            if self.idf_schur is not None:
                self.idf_schur.linearize(X.primal, state)

            # move the vector to the RHS
            kkt_rhs.equals(dLdX)
            kkt_rhs.times(-1.)

            # reset the primal-dual step vector
            P.equals(0.0)

            # trigger the krylov solution
            self.krylov.solve(self.mat_vec, kkt_rhs, P, self.precond)
            self.radius = self.krylov.radius

            # apply globalization
            if self.globalization == 'trust':
                old_flag = min_radius_active
                success, min_radius_active = self.trust_step(
                    X, state, adjoint, P, kkt_rhs,
                    state_work, dual_work, kkt_work, kkt_save)

                # watchdog on trust region failures
                if min_radius_active and old_flag:
                    self.info_file.write(
                        'Trust radius breakdown! Terminating...\n')
                    break
            elif self.globalization == 'filter':
                old_flag = min_radius_active
                success, min_radius_active = self.filter_step(
                    X, state, P, kkt_rhs, kkt_work, state_work, dual_work)

                # watchdog on trust region failures
                if min_radius_active and old_flag:
                    self.info_file.write(
                        'Trust radius breakdown! Terminating...\n')
                    break

                # if filter was successful, compute adjoint for next iteration
                if success:
                    if self.factor_matrices and self.iter < self.max_iter:
                        factor_linear_system(X.primal, state)
                    adjoint.equals_lagrangian_adjoint(X, state, state_work)
            else:
                # accept step
                X.primal.plus(P.primal)
                X.dual.plus(P.dual)

                # calculate states
                state.equals_primal_solution(X.primal)

                # if this is a matrix-based problem, tell the solver to factor
                # some important matrices to be used in the next iteration
                if self.factor_matrices and self.iter < self.max_iter:
                    factor_linear_system(X.primal, state)

                # perform an adjoint solution for the Lagrangian
                adjoint.equals_lagrangian_adjoint(X, state, state_work)

            # send current solution info to the user
            solver_info = current_solution(
                self.iter, X.primal, state, adjoint, X.dual)
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

    def filter_step(self, X, state, P, kkt_rhs, kkt_work, state_work, dual_work):
        filter_success = False
        min_radius_active = False
        max_filter_iter = 3
        for i in xrange(max_filter_iter):
            self.info_file.write('\nFilter Step : iter %i\n'%(i + 1))
            X.plus(P)
            state_work.equals(state)  # save state for reverting later

            solve_failed = False
            if state.equals_primal_solution(X.primal):
                obj = objective_value(X.primal, state)
                dual_work.equals_constraints(X.primal, state)
                cnstr_norm = dual_work.norm2
                if self.filter.dominates(obj, cnstr_norm):
                    # point is acceptable so just check radius and break out
                    self.info_file.write(
                        '   New point accepted!\n')
                    if i == 0 and self.krylov.trust_active and self.radius < self.max_radius:
                        self.radius = min(2*self.radius, self.max_radius)
                        self.krylov.radius = self.radius
                        self.info_file.write(
                            '   Radius increased -> %f\n'%self.radius)
                    filter_success = True
                    break
                else:
                    # point is not acceptable, revert step
                    X.minus(P)
                    state.equals(state_work)
            else:
                # state solve failed, revert step
                self.info_file.write(
                    '   State-solve failed!\n')
                solve_failed = True
                X.minus(P)
                state.equals(state_work)

            # at the first iteration, try a second-order correction
            if i == 0 and not solve_failed:
                kkt_work.equals(P)
                self.info_file.write(
                    '   Attempting a second order correction...')
                self.krylov.apply_correction(dual_work, P)
                X.plus(P)
                if state.equals_primal_solution(X.primal):
                    obj = objective_value(X.primal, state)
                    dual_work.equals_constraints(X.primal, state)
                    cnstr_norm = dual_work.norm2
                    if self.filter.dominates(obj, cnstr_norm):
                        # point is acceptable so just break out
                        self.info_file.write(
                            'SUCCESS!\n')
                        self.info_file.write(
                            '   New point accepted!\n')
                        filter_success = True
                        break
                    else:
                        # point is still not acceptable, revert step
                        X.minus(P)
                        state.equals(state_work)
                        self.info_file.write(
                            'FAILED!\n')
                else:
                    # state solve failed, revert step
                    self.info_file.write(
                        '   State-solve failed!\n')
                    X.minus(P)
                    state.equals(state_work)

            # if we got here, filter has dominated the point
            self.info_file.write('   New point rejected!\n')
            # shrink radius and re-solve
            if self.radius == self.min_radius:
                self.info_file.write('   Reached minimum radius! Exiting filter...\n')
                min_radius_active = True
                break
            else:
                self.radius = max(0.25*self.radius, self.min_radius)
                self.info_file.write('   Radius reduced -> %f!\n'%self.radius)
                self.krylov.radius = self.radius
                self.info_file.write('   Re-solving step...\n')
                self.krylov.re_solve(kkt_rhs, P)

        self.info_file.write('\n')
        return filter_success, min_radius_active

    def trust_step(self, X, state, adjoint, P, kkt_rhs,
                   state_work, dual_work, kkt_work, kkt_save):
        # start trust region loop
        max_iter = 6
        iters = 0
        min_radius_active = False
        converged = False
        self.info_file.write('\n')
        while iters <= max_iter:
            iters += 1
            # evaluate the constraint term at the current step
            dual_work.equals_constraints(X.primal, state)
            # compute the merit value at the current step
            merit_init = objective_value(X.primal, state) \
                         + X.dual.inner(dual_work) \
                         + 0.5*self.mu*(dual_work.norm2**2)
            # add the FLECS step
            kkt_work.equals_ax_p_by(1., X, 1., P)
            # solve states at the new step
            if state_work.equals_primal_solution(kkt_work.primal):
                # evaluate the constraint terms at the new step
                dual_work.equals_constraints(
                    kkt_work.primal, state_work)
                # compute the merit value at the next step
                merit_next = objective_value(kkt_work.primal, state) \
                             + X.dual.inner(dual_work) \
                             + 0.5*self.mu*(dual_work.norm2**2)
                # evaluate the quality of the FLECS model
                rho = (merit_init - merit_next)/self.krylov.pred_aug
            else:
                merit_next = np.nan
                rho = np.nan

            self.info_file.write(
                'Trust Region Step : iter %i\n'%iters +
                '   primal_step    = %e\n'%P.primal.norm2 +
                '   lambda_step    = %e\n'%P.dual.norm2 +
                '\n' +
                '   merit_init     = %e\n'%merit_init +
                '   merit_next     = %e\n'%merit_next +
                '   pred_aug       = %e\n'%self.krylov.pred_aug +
                '   rho            = %e\n'%rho)

            # modify radius based on model quality
            if rho <= 0. or np.isnan(rho):
                # model is bad! -- first we try a 2nd order correction
                if iters == 1:
                    # save the old step in case correction fails
                    kkt_save.equals(P)
                    # attempt a 2nd order correction
                    self.info_file.write(
                        '   Attempting a second order correction...\n')
                    self.krylov.apply_correction(dual_work, P)
                elif iters == 2:
                    # if we got here, the second order correction failed
                    # reject step
                    self.info_file.write(
                        '   Correction failed! Resetting step...\n')
                    P.equals(kkt_save)
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
                        self.krylov.radius = self.radius
                        self.krylov.re_solve(kkt_rhs, P)
                        # self.radius = self.krylov.radius
            else:
                if iters == 2:
                    # 2nd order correction worked -- yay!
                    self.info_file.write('   Correction worked!\n')

                # model is okay -- accept primal step
                self.info_file.write('\nStep accepted!\n')

                # accept the new step entirely
                X.plus(P)
                state.equals_primal_solution(X.primal)

                # if this is a matrix-based problem, tell the solver to factor
                # some important matrices to be used in the next iteration
                if self.factor_matrices and self.iter < self.max_iter:
                    factor_linear_system(X.primal, state)

                # perform an adjoint solution for the Lagrangian
                adjoint.equals_lagrangian_adjoint(X, state, state_work)

                # check the trust radius
                if self.krylov.trust_active:
                    # if active, decide if we want to increase it
                    self.info_file.write('Trust radius active...\n')
                    if rho >= 0.5:
                        # model is good enough -- increase radius
                        self.radius = min(
                            max(2.*P.primal.norm2, self.min_radius),
                            self.max_radius)
                        # self.radius = min(2.*self.radius, self.max_radius)
                        self.info_file.write(
                            '   Radius increased -> %f\n'%self.radius)
                        min_radius_active = False

                # trust radius globalization worked, break loop
                converged = True
                self.info_file.write('\n')
                break

        return converged, min_radius_active


# imports here to prevent circular errors
import numpy as np
from copy import deepcopy
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, objective_value
from kona.linalg.common import factor_linear_system
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.matrices.preconds import ReducedSchurPreconditioner
from kona.linalg.solvers.krylov import FLECS
from kona.linalg.solvers.util import EPS
from kona.algorithms.util.filter import SimpleFilter
