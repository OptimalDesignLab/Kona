import copy
from numpy import sqrt

from kona.options import BadKonaOption, get_opt

from kona.linalg import current_solution, factor_linear_system, objective_value

from kona.linalg.vectors.composite import ReducedKKTVector

from kona.linalg.matrices.common import dCdU, dCdX, dRdU, dRdX
from kona.linalg.matrices.common import IdentityMatrix, ActiveSetMatrix

from kona.linalg.matrices.hessian import IneqCnstrReducedKKTMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix

from kona.linalg.matrices.preconds import NestedKKTPreconditioner
from kona.linalg.matrices.preconds import ReducedSchurPreconditioner

from kona.linalg.solvers.krylov import FLECS

from kona.algorithms.base_algorithm import OptimizationAlgorithm

from kona.algorithms.util import Filter
from kona.algorithms.util.linesearch import BackTracking
from kona.algorithms.util.merit import AugmentedLagrangian

class EqualityConstrainedRSNK(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory, dual_factory, optns={}):
        # trigger base class initialization
        super(EqualityConstrainedRSNK, self).__init__(
            primal_factory, state_factory, dual_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(7)
        self.state_factory.request_num_vectors(5)
        self.dual_factory.request_num_vectors(5)

        # get other options
        self.radius = get_opt(optns, 0.5, 'trust', 'init_radius')
        self.min_radius = get_opt(optns, 0.5/(2**3), 'trust', 'min_radius')
        self.max_radius = get_opt(optns, 0.5*(2**3), 'trust', 'max_radius')
        self.mu_init = get_opt(optns, 1.0, 'aug_lag', 'mu_init')
        self.mu_pow = get_opt(optns, 0.5, 'aug_lag', 'mu_pow')
        self.mu_max = get_opt(optns, 1e5, 'aug_lag', 'mu_max')
        self.ceq_tol = get_opt(optns, 1e-8, 'constraint_tol')
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
        merit_func = get_opt(
            optns, AugmentedLagrangian, 'merit_function', 'type')
        if merit_func not in [AugmentedLagrangian]:
            raise BadKonaOption(optns, 'merit_function', 'type')
        self.merit_func = merit_func(
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

        if self.precond is None:
            # use identity matrix product as preconditioner
            self.eye = IdentityMatrix()
            self.precond = self.eye.product

        elif self.precond == 'nested':
            # for the nested preconditioner we need a new FLECS solver
            # we want this embedded solver to be "silent", therefore we modify
            # the output file location accordingly
            embedded_out = copy.deepcopy(self.info_file)
            embedded_out.file = None
            krylov_optns['out_file'] = embedded_out
            embedded_krylov = krylov(
                [self.primal_factory, self.dual_factory],
                krylov_optns)
            # initialize the nested preconditioner
            self.nested = NestedKKTPreconditioner(
                self.KKT_matrix, embedded_krylov)
            # define preconditioner as approximate solve of KKT system
            self.precond = self.nested.product

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
        dual = self.dual_factory.generate()
        return ReducedKKTVector(primal, dual)

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
        dLdX_save = self._generate_KKT_vector()

        # generate primal vectors
        init_design = self.primal_factory.generate()
        primal_work = []
        for i in xrange(2):
            primal_work.append(self.primal_factory.generate())

        # generate state vectors
        state = self.state_factory.generate()
        state_save = self.state_factory.generate()
        adjoint = self.state_factory.generate()
        state_work = self.state_factory.generate()
        adjoint_work = self.state_factory.generate()

        # generate dual vectors
        dual_work = self.dual_factory.generate()

        # initialize basic data for outer iterations
        converged = False
        self.iter = 0

        # evaluate the initial design before starting outer iterations
        X.equals_init_guess()
        init_design.equals(X._primal)
        state.equals_primal_solution(init_design)
        if self.factor_matrices and self.iter < self.max_iter:
            factor_linear_system(X._primal, state)
        adjoint.equals_adjoint_solution(init_design, state, state_work)
        current_solution(init_design, state, adjoint, X._dual, self.iter)

        # BEGIN BIG LOOP HERE
        ###############################
        for i in xrange(self.max_iter):
            # advance iteration counter
            self.iter += 1

            self.info_file.write(
                '\n' +
                '==========================================\n' +
                'Beginning Major Iteration %i\n'%self.iter +
                '\n')

            # evaluate optimality, feasibility and KKT norms
            dLdX.equals_KKT_conditions(X, state, adjoint, primal_work[0])
            state_save.equals(state)
            if self.iter == 1:
                # calculate initial norms
                grad_norm0 = dLdX._primal.norm2
                feas_norm0 = dLdX._dual.norm2
                kkt_norm0 = sqrt(feas_norm0**2 + grad_norm0**2)
                # set current norms to initial
                kkt_norm = kkt_norm0
                grad_norm = grad_norm0
                feas_norm = feas_norm0
                # print out convergence norms
                self.info_file.write(
                    'grad_norm0 = %e\n'%grad_norm0 +
                    'feas_norm0 = %e\n'%feas_norm0
                )
                # calculate convergence tolerances
                grad_tol = self.primal_tol
                feas_tol = self.ceq_tol * max(feas_norm0, 1e-8)
            else:
                # calculate current norms
                grad_norm = dLdX._primal.norm2
                feas_norm = dLdX._dual.norm2
                kkt_norm = sqrt(feas_norm**2 + grad_norm**2)
                # update the augmented Lagrangian penalty
                self.info_file.write(
                    'grad_norm = %e (%e <-- tolerance)\n'%(
                        grad_norm, grad_tol) +
                    'feas_norm = %e (%e <-- tolerance)\n'%(
                        feas_norm, feas_tol)
                )
            # save the current dL/dX
            dLdX_save.equals(dLdX)

            # write convergence history
            obj_val = objective_value(X._primal, state)
            self._write_history(grad_norm, feas_norm, obj_val)

            # check for convergence
            if (grad_norm < grad_tol) and (feas_norm < feas_tol):
                converged = True
                break

            # compute krylov tolerances in order to achieve superlinear
            # convergence but to avoid oversolving
            krylov_tol = self.krylov.rel_tol*min(1.0, sqrt(kkt_norm/kkt_norm0))
            krylov_tol = max(krylov_tol,
                             min(grad_tol/grad_norm, feas_tol/feas_norm))
            krylov_tol *= self.nu
            self.info_file.write('krylov tol = %e\n'%krylov_tol)

            # set ReducedKKTMatrix product tolerances
            if self.KKT_matrix.dynamic_tol:
                raise NotImplementedError(
                    'EqualityConstrainedRSNK.solve()' +
                    'not yet set up for dynamic tolerance in product')
            else:
                self.KKT_matrix.product_fac *= krylov_tol/self.krylov.max_iter

            # set other solver and product options
            self.KKT_matrix.lamb = 0.0
            self.krylov.rel_tol = krylov_tol
            self.krylov.radius = self.radius
            self.krylov.mu = self.mu

            self.krylov.out_file.write(
                '#-------------------------------------------------\n' +
                '# primal solve\n')

            # reset the primal-dual step vector
            P.equals(0.0)

            # move dL/dX to right hand side
            dLdX.times(-1.)

            # linearize the KKT matrix
            self.KKT_matrix.linearize(X, state, adjoint)
            if self.idf_schur is not None:
                self.idf_schur.linearize(X, state)

            # trigger the krylov solution
            self.krylov.solve(self.mat_vec, dLdX, P, self.precond)

            # move dL/dX back to left hand side
            dLdX.times(-1.)

            # use a trust region algorithm for globalization
            self.trust_step(
                X, state, P, dLdX, krylov_tol, feas_tol,
                primal_work, state_work, adjoint_work, dual_work)

            # if this is a matrix-based problem, tell the solver to factor
            # some important matrices to be used in the next iteration
            if self.factor_matrices and self.iter < self.max_iter:
                factor_linear_system(X._primal, state)

            # solve for adjoint
            adjoint.equals_adjoint_solution(X._primal, state, state_work)

            # write current solution
            current_solution(X._primal, state, adjoint, X._dual, self.iter)
            self.info_file.write('Norm of multipliers = %e\n'%X._dual.norm2)

        ############################
        # END OF BIG LOOP

        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Failed to converge!\n')

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n'%self.iter)

    def trust_step(self, X, state, P, dLdX, krylov_tol, feas_tol,
                   primal_work, state_work, adjoint_work, dual_work):
        # first we have to compute the total derivative of the
        # augmented Lagrangian merit function
        # to do this, we have to solve a special adjoint system
        # STEP 1: define the RHS for the system
        dual_work.equals(dLdX._dual)
        dCdU(X._primal, state).T.product(dual_work, state_work)
        state_work.times(-self.mu)
        # STEP 2: solve the adjoint system
        dRdU(X._primal, state).T.solve(state_work, adjoint_work)
        # STEP 3: with the adjoint, we assemble the gradient
        dCdX(X._primal, state).T.product(dual_work, primal_work[0])
        primal_work[0].times(self.mu)
        dRdX(X._primal, state).T.product(
            adjoint_work, primal_work[1])
        primal_work[0].plus(primal_work[1])
        primal_work[0].plus(dLdX._primal)

        # move dL/dX to RHS for re-solves
        dLdX.times(-1)
        # start trust region loop
        max_iter = 10
        iters = 0
        while iters <= max_iter:
            iters += 1
            # reset merit function at new step
            p_dot_grad = primal_work[0].inner(P._primal)
            self.merit_func.reset(X, state, P, p_dot_grad, self.mu)
            # evaluate the quality of the FLECS model
            merit_init = self.merit_func.func_val
            merit_next = self.merit_func.eval_func(1.)
            rho = (merit_init - merit_next)/self.krylov.pred_aug
            self.info_file.write(
                'Trust Region : iter %i\n'%iters +
                '   merit_init = %f\n'%merit_init +
                '   merit_next = %f\n'%merit_next +
                '   pred_aug = %f\n'%self.krylov.pred_aug +
                '   rho = %f\n'%rho)

            # modify radius based on model quality
            if rho < 0.01:
                # model is bad -- shrink radius and re-solve
                self.radius = max(0.5*self.radius, self.min_radius)
                self.info_file.write(
                    '   Re-solving with new radius -> %f\n'%self.radius)
                self.krylov.radius = self.radius
                self.krylov.re_solve(dLdX, P)
            else:
                # model is okay -- accept step
                self.info_file.write('Step accepted!\n')
                X.plus(P)
                state.equals_primal_solution(X._primal)
                if rho > 0.75 and self.krylov.trust_active:
                    # model is exceptionally good -- increase radius
                    self.radius = min(2*self.radius, self.max_radius)
                    self.info_file.write(
                        '   Radius increased -> %f\n'%self.radius)
                # update the penalty parameter
                dual_work.equals_constraints(X._primal, state)
                cnstr_norm = dual_work.norm2
                if cnstr_norm > self.eta:
                    self.mu = min(10.**self.mu_pow, self.mu_max)
                    self.eta = 1./(self.mu**0.1)
                    self.info_file.write('   New mu = %1.2e\n'%self.mu)
                else:
                    if cnstr_norm > feas_tol:
                        self.eta = self.eta/(self.mu**0.9)
                break

class InequalityConstrainedRSNK(EqualityConstrainedRSNK):

    def __init__(self, primal_factory, state_factory, dual_factory, optns={}):
        super(InequalityConstrainedRSNK, self).__init__(
            primal_factory, state_factory, dual_factory, optns
        )

        self.KKT_matrix = IneqCnstrReducedKKTMatrix(self.KKT_matrix)
        self.mat_vec = self.KKT_matrix.product

        if get_opt(optns, None, 'reduced', 'precond') == 'nested':
            self.nested.KKT_matrix = self.KKT_matrix

    def trust_step(self, X, state, P, dLdX, krylov_tol, feas_tol,
                   primal_work, state_work, adjoint_work, dual_work):
        ActiveSetMatrix(dLdX._dual).product(dLdX._dual, dLdX._dual)
        super(InequalityConstrainedRSNK, self).trust_step(
            X, state, P, dLdX, krylov_tol, feas_tol,
            primal_work, state_work, adjoint_work, dual_work)
