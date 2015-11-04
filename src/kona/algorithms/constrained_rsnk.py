import copy
import warnings
import numpy as np

from kona.options import BadKonaOption, get_opt
from kona.linalg import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.common import dCdX, dCdU, dRdX, dRdU, IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.matrices.preconds import NestedKKTPreconditioner
from kona.linalg.matrices.preconds import ReducedSchurPreconditioner
from kona.linalg.solvers.krylov import FLECS, FGMRES
from kona.linalg.solvers.util import EPS
from kona.algorithms.base_algorithm import OptimizationAlgorithm
from kona.algorithms.util.merit import AugmentedLagrangian

class ConstrainedRSNK(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory, dual_factory, optns={}):
        # trigger base class initialization
        super(ConstrainedRSNK, self).__init__(
            primal_factory, state_factory, dual_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(6)
        self.state_factory.request_num_vectors(4)
        self.dual_factory.request_num_vectors(11)

        # get other options
        self.radius = get_opt(optns, 0.5, 'trust', 'init_radius')
        self.min_radius = get_opt(optns, 0.5/(2**3), 'trust', 'min_radius')
        self.max_radius = get_opt(optns, 0.5*(2**3), 'trust', 'max_radius')
        self.mu = get_opt(optns, 1.0, 'aug_lag', 'mu_init')
        self.mu_pow = get_opt(optns, 0.5, 'aug_lag', 'mu_pow')
        self.mu_max = get_opt(optns, 1e5, 'aug_lag', 'mu_max')
        self.cnstr_tol = get_opt(optns, 1e-8, 'constraint_tol')
        self.nu = get_opt(optns, 0.95, 'reduced', 'nu')
        self.factor_matrices = get_opt(optns, False, 'matrix_explicit')

        # set the krylov solver
        acceptable_solvers = [FLECS]
        try:
            krylov = get_opt(optns, FLECS, 'krylov', 'solver')
            if krylov not in acceptable_solvers:
                raise BadKonaOption(optns, 'krylov', 'solver')
            krylov_optns = get_opt(optns, {}, 'krylov')
            self.krylov = krylov(
                [self.primal_factory, self.dual_factory],
                krylov_optns)
        except Exception:
            raise BadKonaOption(optns, 'krylov', 'solver')

        # penalty parameter data
        self.eta = 1./(self.mu**0.1)

        # initialize the globalization method
        # NOTE: Latest C++ source has the filter disabled entirely!!!
        # set the type of line-search algorithm
        self.merit_func = AugmentedLagrangian(
            self.primal_factory, self.state_factory, self.dual_factory,
            {}, self.info_file)

        # initialize the KKT matrix definition
        reduced_optns = get_opt(optns, {}, 'reduced')
        reduced_optns['out_file'] = self.info_file
        self.KKT_matrix = ReducedKKTMatrix(
            [self.primal_factory, self.state_factory, self.dual_factory],
            reduced_optns)
        self.mat_vec = self.KKT_matrix.product

        # initialize the preconditioner for the KKT matrix
        self.precond = get_opt(optns, None, 'reduced', 'precond')
        self.idf_schur = None
        self.nested = None

        if self.precond is None:
            # use identity matrix product as preconditioner
            self.eye = IdentityMatrix()
            self.precond = self.eye.product

        elif self.precond == 'nested':
            # for the nested preconditioner we need a new FLECS solver
            # we want this embedded solver to be "silent", therefore we modify
            # the output file location accordingly
            embedded_out = copy.deepcopy(self.info_file)
            embedded_out.file = self.primal_factory._memory.open_file(
                'kona_nested.dat')
            krylov_optns['out_file'] = embedded_out
            krylov_optns['max_iter'] = 100
            # embedded_krylov = krylov(
            #     [self.primal_factory, self.dual_factory],
            #     krylov_optns)
            embedded_krylov = FGMRES(
                self.primal_factory,
                optns=krylov_optns,
                dual_factory=self.dual_factory)
            # initialize the nested preconditioner
            self.nested = NestedKKTPreconditioner(
                [self.primal_factory, self.state_factory, self.dual_factory],
                reduced_optns)
            # set in the krylov object for the nested solve
            self.nested.set_krylov_solver(embedded_krylov)
            self.nested.krylov.rel_tol = 1e-12
            # define preconditioner as a nested solution of the approximate KKT
            self.precond = self.nested.solve

        elif self.precond == 'idf_schur':
            self.idf_schur = ReducedSchurPreconditioner(
                [self.primal_factory, self.state_factory, self.dual_factory])
            self.precond = self.idf_schur.product

        else:
            raise BadKonaOption(optns, 'reduced', 'precond')

    def _write_header(self):
        self.hist_file.write(
            '# Kona equality-constrained RSNK convergence history file\n' +
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
            '#%6i'%self.iter + ' '*5 +
            '%7i'%self.primal_factory._memory.cost + ' '*5 +
            '%11e'%opt + ' '*5 +
            '%11e'%feas + ' '*5 +
            '%11e'%obj + ' '*5 +
            '%11e'%self.mu + ' '*5 +
            '%11e'%self.radius + '\n'
        )

    def _generate_KKT_vector(self):
        design = self.primal_factory.generate()
        slack = self.dual_factory.generate()
        primal = CompositePrimalVector(design, slack)
        dual = self.dual_factory.generate()
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
        kkt_work = self._generate_KKT_vector()

        # generate primal vectors
        primal_work = self.primal_factory.generate()
        primal_trial = self.primal_factory.generate()

        # generate state vectors
        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        adjoint = self.state_factory.generate()
        adjoint_work = self.state_factory.generate()

        # generate dual vectors
        dual_work = self.dual_factory.generate()
        slack_work = self.dual_factory.generate()
        slack_trial = self.dual_factory.generate()

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
        if solver_info is not None:
            self.info_file.write(solver_info)

        # BEGIN NEWTON LOOP HERE
        ###############################
        min_radius_active = False
        for i in xrange(self.max_iter):
            # advance iteration counter
            self.iter += 1

            # evaluate optimality, feasibility and KKT norms
            dLdX.equals_KKT_conditions(
                X, state, adjoint, primal_work, dual_work)
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
                    'feas_norm0         = %e\n'%feas_norm0
                )
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
                    'feas_norm          = %e (%e <-- tolerance)\n'%(
                        feas_norm, feas_tol)
                )

            # write convergence history
            obj_val = objective_value(X._primal._design, state)
            self._write_history(grad_norm, feas_norm, obj_val)

            # check for convergence
            if (grad_norm < grad_tol) and (feas_norm < feas_tol):
                converged = True
                break

            # compute krylov tolerances in order to achieve superlinear
            # convergence but to avoid oversolving
            krylov_tol = self.krylov.rel_tol*min(
                1.0, np.sqrt(kkt_norm/kkt_norm0))
            krylov_tol = max(krylov_tol,
                             min(grad_tol/grad_norm,
                                 feas_tol/feas_norm))
            krylov_tol *= self.nu

            # set ReducedKKTMatrix product tolerances
            if self.KKT_matrix.dynamic_tol:
                raise NotImplementedError(
                    'ConstrainedRSNK.solve()' +
                    'not yet set up for dynamic tolerance in product')
            else:
                self.KKT_matrix.product_fac *= \
                    krylov_tol/self.krylov.max_iter

            # set other solver and product options
            self.KKT_matrix.lamb = 0.0
            self.krylov.rel_tol = krylov_tol
            self.krylov.radius = self.radius
            self.krylov.mu = self.mu

            # linearize the KKT matrix
            self.KKT_matrix.linearize(X, state, adjoint)

            # propagate options through the preconditioners
            if self.idf_schur is not None:
                self.idf_schur.linearize(X, state)

            if self.nested is not None:
                self.nested.linearize(X, state, adjoint)

            # form the RHS vector for the KKT system
            # the primal component is equal to the objective derivative
            state_work.equals_objective_partial(X._primal._design, state)
            state_work.times(-1.)
            dRdU(X._primal._design, state).T.solve(state_work, adjoint)
            kkt_rhs._primal._design.equals_total_gradient(
                X._primal._design, state, adjoint, primal_work)
            # the slack component of the RHS vector is zero
            kkt_rhs._primal._slack.equals(0.0)
            # the dual component is equal to the KKT condition (c - e^s)
            kkt_rhs._dual.equals(dLdX._dual)

            # move the vector to the RHS
            kkt_rhs.times(-1.)

            # reset the primal-dual step vector
            P.equals(0.0)

            # trigger the krylov solution
            self.krylov.solve(self.mat_vec, kkt_rhs, P, self.precond)

            # apply globalization
            linesearch = True
            trust_region = False

            old_flag = min_radius_active
            if linesearch:
                success, min_radius_active = self.backtracking_step(
                    X, state, P, dLdX, primal_trial, slack_trial,
                    primal_work, state_work, adjoint_work,
                    dual_work, slack_work, feas_tol)
            elif trust_region:
                success, min_radius_active = self.trust_step(
                    X, state, P, kkt_rhs, krylov_tol, feas_tol,
                    state_work, dual_work, slack_work, kkt_work)
            else:
                X._primal.plus(P._primal)
                X._dual.equals(P._dual)
                state.equals_primal_solution(X._primal._design)

            # watchdog on trust region failures
            if min_radius_active and old_flag:
                self.info_file.write(
                    'Trust radius breakdown! Terminating...\n')
                break

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
            if solver_info is not None:
                self.info_file.write(solver_info)

        ############################
        # END OF NEWTON LOOP

        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Optimization FAILED!\n')

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n\n'%self.iter)

    def backtracking_step(self, X, state, P, dLdX, design_trial, slack_trial,
                          primal_work, state_work, adjoint_work,
                          dual_work, slack_work, feas_tol):
        # do some aliasing to make the code cleanier
        design_vars = X._primal._design
        design_step = P._primal._design
        slack_vars = X._primal._slack
        slack_step = P._primal._slack

        # first we have to compute the total derivative of the
        # merit function w.r.t. design variables
        # STEP 1: evaluate the total derivative of the objective function
        state_work.equals_objective_partial(design_vars, state)
        state_work.times(-1.)
        dRdU(design_vars, state).T.solve(state_work, adjoint_work)
        primal_work.equals_total_gradient(
            design_vars, state, adjoint_work, design_trial)
        # STEP 2: evaluate the total derivative of the penalty term
        dual_work.equals_constraints(design_vars, state)
        slack_work.exp(slack_vars)
        slack_work.times(-1.)
        slack_work.restrict()
        dual_work.plus(slack_work)
        dCdU(design_vars, state).T.product(dual_work, state_work)
        state_work.times(-self.mu)
        dRdU(design_vars, state).T.solve(state_work, adjoint_work)
        dCdX(design_vars, state).T.product(dual_work, design_trial)
        design_trial.times(self.mu)
        primal_work.plus(design_trial)
        dRdX(design_vars, state).T.product(
            adjoint_work, design_trial)
        primal_work.plus(design_trial)

        # now compute the slack derivative of the merit function
        dual_work.equals_constraints(design_vars, state)
        slack_work.exp(slack_vars)
        dual_work.times(slack_work)
        dual_work.times(-1.)
        dual_work.plus(slack_work)
        dual_work.times(self.mu)
        dual_work.restrict()

        # calculate scalar gradient in the descent direction
        p_dot_grad = \
            primal_work.inner(design_step) + dual_work.inner(slack_step)
        if p_dot_grad > 0.:
            raise ValueError('FLECS step is not a descent direction!')

        # calculate initial merit function
        dual_work.equals_constraints(design_vars, state)
        slack_work.exp(slack_vars)
        slack_work.times(-1.)
        slack_work.restrict()
        dual_work.plus(slack_work)
        f_init = objective_value(design_vars, state)
        f_init += 0.5*self.mu*(dual_work.norm2**2)

        # set some line search parameters
        alpha = 1.
        alpha_min = 1e-4
        rdtn_factor = 0.5
        dec_cond = 1e-4
        max_iter = 10
        iters = 0

        # start backtracking line search
        min_radius_active = False
        converged = False
        # design_trial.equals(design_vars)
        # slack_trial.equals(slack_vars)
        self.info_file.write(
            '\nEntering Line-search...\n' +
            '   design_step    = %e\n'%design_step.norm2 +
            '   slack_step     = %e\n'%slack_step.norm2 +
            '   new lambda     = %e\n'%P._dual.norm2 +
            '   merit_init     = %e\n'%f_init)
        while iters < max_iter:
            iters += 1

            # take a design step
            design_trial.equals_ax_p_by(1., design_vars, alpha, design_step)
            # take a scaled slack step
            slack_trial.equals_ax_p_by(1., slack_vars, alpha, slack_work)
            # solve for states
            state_work.equals_primal_solution(design_trial)

            # calculate the constraint term
            dual_work.equals_constraints(design_trial, state_work)
            slack_work.exp(slack_trial)
            slack_work.times(-1.)
            slack_work.restrict()
            dual_work.plus(slack_work)

            f_trial = objective_value(design_trial, state_work)
            f_trial += 0.5*self.mu*(dual_work.norm2**2)

            f_sufficient = f_init + dec_cond*alpha*p_dot_grad

            self.info_file.write(
                '   Iteration  %i :\n'%iters +
                '      alpha       = %e\n'%alpha +
                '      merit_suff  = %e\n'%f_sufficient +
                '      merit_next  = %e\n'%f_trial)

            if f_trial <= f_sufficient:
                # sufficient decrease satisdied, accept step!
                self.info_file.write(
                    '\nStep accepted!\n')
                X._primal._design.equals(design_trial)
                X._primal._slack.plus(slack_step)
                X._dual.equals(P._dual)

                # evaluate constraints at the new point
                dual_work.equals_constraints(X._primal._design, state)
                slack_work.exp(X._primal._slack)
                slack_work.times(-1.)
                slack_work.restrict()
                dual_work.plus(slack_work)
                feas_norm = dual_work.norm2

                # update the penalty coefficient
                if feas_norm <= feas_tol:
                    self.info_file.write('Feasibility satisfied!')
                else:
                    if feas_norm <= self.eta:
                        # constraints are good, tighten tolerance
                        self.eta = self.eta/(self.mu**0.9)
                    else:
                        # constraints are bad, increase the penalty parameter
                        self.mu = min(self.mu*(10.**self.mu_pow), self.mu_max)
                        self.eta = 1./(self.mu**0.1)
                        self.info_file.write(
                            '   Mu increased -> %1.2e\n'%self.mu)

                # increase radius
                if self.krylov.trust_active:
                    self.radius = min(2*alpha*self.radius, self.max_radius)
                    self.info_file.write(
                        '   New radius -> %f\n'%self.radius)

                # flag convergence and break loop
                converged = True
                break

            else:
                # sufficient decrease not satisifed, shrink alpha
                if alpha == alpha_min:
                    self.info_file.write(
                        '\nMinimum alpha reached! Terminating...\n')
                    break
                else:
                    alpha = max(alpha*rdtn_factor, alpha_min)

        # if line-search failed, shrink the trust radius for FLECS
        if not converged:
            self.radius = max(self.radius/2, self.min_radius)
            self.info_file.write(
                'Line search failed...\n' +
                '   Radius shrunk -> %f\n'%self.radius)
            if self.radius == self.min_radius:
                min_radius_active = True

        return converged, min_radius_active

    def trust_step(self, X, state, P, kkt_rhs, krylov_tol, feas_tol,
                   state_work, dual_work, slack_work, kkt_work):
        # start trust region loop
        max_iter = 6
        iters = 0
        min_radius_active = False
        converged = False
        self.info_file.write('\n')
        while iters <= max_iter:
            iters += 1
            # reset merit function at new step
            self.merit_func.reset(
                X, state, P, 0.0, self.mu)
            # evaluate the merit ats the current step
            merit_init = self.merit_func.func_val
            # evaluate the merit at the next step
            merit_failed = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    merit_next = self.merit_func.eval_func(1.)
                except Exception:
                    merit_failed = True
            # evaluate the quality of the FLECS model
            rho = (merit_init - merit_next)/max(self.krylov.pred_aug, EPS)

            self.info_file.write(
                'Trust Region Step : iter %i\n'%iters +
                '   design_step    = %e\n'%P._primal._design.norm2 +
                '   slack_step     = %e\n'%P._primal._slack.norm2 +
                '   new lambda     = %e\n'%P._dual.norm2 +
                '\n' +
                '   merit_init     = %e\n'%merit_init +
                '   merit_next     = %e\n'%merit_next +
                '   pred_aug       = %e\n'%self.krylov.pred_aug +
                '   rho            = %f\n'%rho)

            # modify radius based on model quality
            if rho <= 0.01 or np.isnan(rho) or merit_failed:
                # model is bad! -- first we try a 2nd order correction
                if iters == 1:
                    # save old design and state points
                    kkt_work.equals(X)
                    state_work.equals(state)
                    # evaluate the constraints at the new point
                    X._primal.plus(P._primal)
                    X._dual.equals(P._dual)
                    state.equals_primal_solution(X._primal._design)
                    dual_work.equals_constraints(X._primal._design, state)
                    slack_work.exp(X._primal._slack)
                    slack_work.times(-1.)
                    slack_work.restrict()
                    dual_work.plus(slack_work)
                    # reset the point back and save the step for later
                    X.equals(kkt_work)
                    state.equals(state_work)
                    kkt_work.equals(P)
                    # attempt a 2nd order correction
                    self.info_file.write(
                        '   Attempting a second order correction...\n')
                    self.krylov.apply_correction(dual_work, P)
                elif iters == 2:
                    # if we got here, the second order correction failed
                    # reject step
                    self.info_file.write(
                        '   Correction failed! Resetting step...\n')
                    P.equals(kkt_work)
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
            else:
                if iters == 2:
                    # 2nd order correction worked -- yay!
                    self.info_file.write('   Correction worked!\n')

                # model is okay -- accept primal step
                self.info_file.write('\nStep accepted!\n')
                X._primal.plus(P._primal)
                X._dual.equals(P._dual)
                state.equals_primal_solution(X._primal._design)

                # evaluate constraints at the new point
                dual_work.equals_constraints(X._primal._design, state)
                slack_work.exp(X._primal._slack)
                slack_work.times(-1.)
                slack_work.restrict()
                dual_work.plus(slack_work)
                feas_norm = dual_work.norm2

                # update the penalty coefficient
                if feas_norm < feas_tol:
                    self.info_file.write('Feasibility satisfied!')
                else:
                    if feas_norm <= self.eta:
                        # constraints are good, tighten tolerance
                        self.eta = self.eta/(self.mu**0.9)
                    else:
                        # constraints are bad, increase the penalty parameter
                        self.mu = min(self.mu*(10.**self.mu_pow), self.mu_max)
                        self.eta = 1./(self.mu**0.1)
                        self.info_file.write(
                            '   Mu increased -> %1.2e\n'%self.mu)

                # check the trust radius
                if self.krylov.trust_active:
                    # if active, decide if we want to increase it
                    self.info_file.write('Trust radius active...\n')
                    if rho > 0.5:
                        # model is good -- increase radius
                        self.radius = min(2.*self.radius, self.max_radius)
                        self.info_file.write(
                            '   Radius increased -> %f\n'%self.radius)
                        min_radius_active = False

                # trust radius globalization worked, break loop
                converged = True
                self.info_file.write('\n')
                break

        return converged, min_radius_active
