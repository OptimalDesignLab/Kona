import copy
import warnings
import numpy as np

from kona.options import BadKonaOption, get_opt
from kona.linalg import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector, DesignSlackComposite
from kona.linalg.matrices.common import dCdX, IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.matrices.preconds import NestedKKTPreconditioner
from kona.linalg.matrices.preconds import ReducedSchurPreconditioner
from kona.linalg.solvers.krylov import FLECS
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
        self.state_factory.request_num_vectors(3)
        self.dual_factory.request_num_vectors(10)

        # get other options
        self.radius = get_opt(optns, 0.5, 'trust', 'init_radius')
        self.min_radius = get_opt(optns, 0.5/(2**3), 'trust', 'min_radius')
        self.max_radius = get_opt(optns, 0.5*(2**3), 'trust', 'max_radius')
        self.mu_init = get_opt(optns, 1.0, 'aug_lag', 'mu_init')
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
        self.mu = self.mu_init
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
            embedded_krylov = krylov(
                [self.primal_factory, self.dual_factory],
                krylov_optns)
            # initialize the nested preconditioner
            self.nested = NestedKKTPreconditioner(
                [self.primal_factory, self.state_factory, self.dual_factory],
                reduced_optns)
            # set in the krylov object for the nested solve
            self.nested.set_krylov_solver(embedded_krylov)
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
        slack = self.dual_factory.generate()
        dual = self.dual_factory.generate()
        return ReducedKKTVector(DesignSlackComposite(primal, slack), dual)

    def solve(self):

        self._write_header()
        self.info_file.write(
            '**************************************************\n' +
            '***        Using FLECS-based Algorithm         ***\n' +
            '**************************************************\n')

        # generate composite KKT vectors
        X = self._generate_KKT_vector()
        P = self._generate_KKT_vector()
        dLdX = self._generate_KKT_vector()
        kkt_work = self._generate_KKT_vector()
        # kkt_rhs = self._generate_KKT_vector()

        # generate primal vectors
        init_design = self.primal_factory.generate()
        primal_work = self.primal_factory.generate()

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
        P.equals(0.0)
        X.equals_init_guess()
        init_design.equals(X._primal._design)
        state.equals_primal_solution(init_design)
        if self.factor_matrices and self.iter < self.max_iter:
            factor_linear_system(X._primal._design, state)
        adjoint.equals_adjoint_solution(init_design, state, state_work)
        current_solution(init_design, state, adjoint, X._dual, self.iter)

        # BEGIN BIG LOOP HERE
        ###############################
        min_radius_active = False
        for i in xrange(self.max_iter):
            # advance iteration counter
            self.iter += 1

            slack_work.equals(X._primal._slack)
            slack_work.exp(slack_work)
            slack_work.restrict()
            self.info_file.write(
                '\n' +
                '==========================================\n' +
                'Beginning Major Iteration %i\n'%self.iter +
                '\n' +
                'design vars norm = %e\n'%X._primal._design.norm2 +
                'slack vars norm  = %e\n'%X._primal._slack.norm2 +
                'slack term norm  = %e\n'%slack_work.norm2 +
                'multiplier norm  = %e\n'%X._dual.norm2 +
                '\n')

            # evaluate optimality, feasibility and KKT norms
            dLdX.equals_KKT_conditions(
                X, state, adjoint, primal_work, dual_work)
            if self.iter == 1:
                # calculate initial norms
                grad_norm0 = dLdX._primal.norm2
                feas_norm0 = max(dLdX._dual.norm2, EPS)
                kkt_norm0 = np.sqrt(feas_norm0**2 + grad_norm0**2)
                # set current norms to initial
                kkt_norm = kkt_norm0
                grad_norm = grad_norm0
                feas_norm = feas_norm0
                # print out convergence norms
                self.info_file.write(
                    'grad_norm0  = %e\n'%grad_norm0 +
                    'feas_norm0  = %e\n'%feas_norm0
                )
                # calculate convergence tolerances
                grad_tol = self.primal_tol * max(grad_norm0, 1e-3)
                feas_tol = self.cnstr_tol * max(feas_norm0, 1e-3)
            else:
                # calculate current norms
                grad_norm = dLdX._primal.norm2
                feas_norm = max(dLdX._dual.norm2, EPS)
                kkt_norm = np.sqrt(feas_norm**2 + grad_norm**2)
                # update the augmented Lagrangian penalty
                self.info_file.write(
                    'grad_norm   = %e (%e <-- tolerance)\n'%(
                        grad_norm, grad_tol) +
                    'feas_norm   = %e (%e <-- tolerance)\n'%(
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
                             min(grad_tol/grad_norm, feas_tol/feas_norm))
            krylov_tol *= self.nu
            self.info_file.write('krylov tol = %e\n\n'%krylov_tol)

            # set ReducedKKTMatrix product tolerances
            if self.KKT_matrix.dynamic_tol:
                raise NotImplementedError(
                    'ConstrainedRSNK.solve()' +
                    'not yet set up for dynamic tolerance in product')
            else:
                self.KKT_matrix.product_fac *= krylov_tol/self.krylov.max_iter

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
                if self.nested.dynamic_tol:
                    raise NotImplementedError(
                        'ConstrainedRSNK.solve()' +
                        'not yet set up for dynamic tolerance in nested solve')
                else:
                    self.nested.product_fac *= krylov_tol/self.krylov.max_iter
                self.nested.lamb = 0.0
                self.nested.krylov.rel_tol = self.krylov.rel_tol
                self.nested.krylov.radius = self.krylov.radius
                self.nested.krylov.mu = self.krylov.mu
                self.nested.linearize(X, state, adjoint)

            # reset the primal-dual step vector
            P.equals(0.0)

            # # manipulate the dLdX to produce the primal problem RHS vector
            # kkt_rhs.equals(dLdX)
            # kkt_rhs.times(-1.)
            # # first we remove the (dC/dX)^T * lambda term from the design deriv
            # dCdX(X._primal._design, state).T.product(X._dual, primal_work)
            # kkt_rhs._primal._design.plus(primal_work)
            # # then we remove lambda^T * diag(e^s) from the slack derivative
            # slack_work.equals(X._primal._slack)
            # slack_work.exp(slack_work)
            # slack_work.restrict()
            # slack_work.times(X._dual)
            # kkt_rhs._primal._slack.plus(slack_work)

            # move dLdX to RHS
            dLdX.times(-1.)

            # trigger the krylov solution
            self.krylov.solve(self.mat_vec, dLdX, P, self.precond)

            # use a trust region algorithm for globalization
            old_flag = min_radius_active
            min_radius_active = self.trust_step(
                X, state, P, dLdX, krylov_tol, feas_tol,
                state_work, dual_work, slack_work, kkt_work)

            # watchdog on trust region failures
            if min_radius_active and old_flag:
                self.info_file.write(
                    'Trust radius breakdown! Terminating...\n')
                break

            # X.plus(P)
            # X._primal._slack.restrict()
            # state.equals_primal_solution(X._primal._design)

            # if this is a matrix-based problem, tell the solver to factor
            # some important matrices to be used in the next iteration
            if self.factor_matrices and self.iter < self.max_iter:
                factor_linear_system(X._primal._design, state)

            # solve for adjoint
            adjoint.equals_adjoint_solution(
                X._primal._design, state, state_work)

            # write current solution
            current_solution(
                X._primal._design, state, adjoint, X._dual, self.iter)

        ############################
        # END OF BIG LOOP

        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Failed to converge!\n')

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n'%self.iter)

    def trust_step(self, X, state, P, kkt_rhs, krylov_tol, feas_tol,
                   state_work, dual_work, slack_work, kkt_work):
        # save old step before any modifications
        kkt_work.equals(P)
        # start trust region loop
        max_iter = 6
        iters = 0
        min_radius_active = False
        while iters <= max_iter:
            iters += 1
            # reset merit function at new step
            self.merit_func.reset(X, state, P, 0.0, self.mu)
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
                'Trust Region Loop : iter %i\n'%iters +
                '   merit_init = %f\n'%merit_init +
                '   merit_next = %f\n'%merit_next +
                '   pred_aug = %f\n'%self.krylov.pred_aug +
                '   rho = %f\n'%rho)

            # modify radius based on model quality
            if rho <= 0.01 or np.isnan(rho) or merit_failed:
                # model is bad! -- first we try a 2nd order correction
                if iters == 1:
                    # save old design and state points
                    kkt_work.equals(X)
                    state_work.equals(state)
                    # evaluate the constraints at the new point
                    X.plus(P)
                    state.equals_primal_solution(X._primal._design)
                    dual_work.equals_constraints(X._primal._design, state)
                    slack_work.exp(X._primal._slack)
                    slack_work.restrict()
                    slack_work.times(-1.)
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
                    self.info_file.write('   Shrinking trust radius...\n')
                    old_radius = self.radius
                    self.radius = max(0.5*self.radius, self.min_radius)
                    if self.radius == old_radius:
                        self.info_file.write(
                            '      Reached minimum radius! ' +
                            'Exiting globalization...\n')
                        min_radius_active = True
                        return min_radius_active
                    else:
                        self.info_file.write(
                            '      Re-solving @ delta = %f\n'%self.radius)
                        self.krylov.radius = self.radius
                        self.krylov.re_solve(kkt_rhs, P)
            else:
                if iters == 2:
                    # 2nd order correction worked -- yay!
                    self.info_file.write('   Correction worked!\n\n')
                # model is okay -- accept primal step
                self.info_file.write('Primal step accepted!\n')
                X._primal.plus(P._primal)
                # X._dual.equals(P._dual)
                state.equals_primal_solution(X._primal._design)
                # evaluate constraints at the new point
                dual_work.equals_constraints(X._primal._design, state)
                slack_work.exp(X._primal._slack)
                slack_work.restrict()
                slack_work.times(-1.)
                dual_work.plus(slack_work)
                feas_norm = dual_work.norm2
                # evaluate feasibility
                if feas_norm > feas_tol:
                    # update the penalty parameter
                    if feas_norm <= self.eta:
                        self.info_file.write('Dual step accepted!\n')
                        X._dual.plus(P._dual)
                        self.eta = self.eta/(self.mu**0.9)
                    else:
                        # constraints aren't good
                        # reject multipliers and increase mu
                        self.info_file.write('Dual step rejected...\n')
                        self.mu = min(self.mu*(10.**self.mu_pow), self.mu_max)
                        self.eta = 1./(self.mu**0.1)
                        self.info_file.write(
                            '   Mu increased -> %1.2e\n'%self.mu)
                else:
                    self.info_file.write('Feasibility satisfied!\n')
                # check the trust radius
                if self.krylov.trust_active:
                    # if active, decide if we want to increase it
                    self.info_file.write('Trust radius active...\n')
                    if rho > 0.75:
                        # model is good -- increase radius
                        self.radius = min(2*self.radius, self.max_radius)
                        self.info_file.write(
                            '   Radius increased -> %f\n'%self.radius)

                # trust radius globalization worked, break loop
                self.info_file.write('\n')
                return min_radius_active
