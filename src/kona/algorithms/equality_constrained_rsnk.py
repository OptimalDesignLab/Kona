import sys, copy
from numpy import sqrt

from kona.options import BadKonaOption, get_opt

from kona.linalg import current_solution, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import dRdU, IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.matrices.preconditioners import ReducedSchurPreconditioner, \
                                                 NestedKKTPreconditioner
from kona.linalg.solvers.krylov import FLECS
from kona.algorithms.util import Filter
from kona.algorithms.base_algorithm import OptimizationAlgorithm

class EqualityConstrainedRSNK(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory, dual_factory, optns={}):
        # trigger base class initialization
        super(EqualityConstrainedRSNK, self).__init__(
            primal_factory, state_factory, dual_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(11)
        self.state_factory.request_num_vectors(9)
        self.dual_factory.request_num_vectors(6)

        # get other options
        self.radius = get_opt(optns, 0.1, 'trust', 'init_radius')
        self.max_radius = get_opt(optns, 1.0, 'trust', 'max_radius')
        self.trust_tol = get_opt(optns, 0.1, 'trust', 'tol')
        self.mu_init = get_opt(optns, 0.1, 'aug_lag', 'mu_init')
        self.mu_pow = get_opt(optns, 1, 'aug_lag', 'mu_pow')
        self.ceq_tol = get_opt(optns, 1e-8, 'contraint_tol')
        self.nu = get_opt(optns, 0.95, 'reduced', 'nu')

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
        except:
            raise BadKonaOption(optns, 'krylov', 'solver')

        # create the filter
        # NOTE: Latest C++ source has the filter disabled entirely!!!
        self.filter = Filter()

        # initialize the KKT matrix definition
        reduced_optns = get_opt(optns, {}, 'reduced')
        reduced_optns['out_file'] = self.info_file
        self.KKT_matrix = ReducedKKTMatrix(
            [self.primal_factory, self.state_factory, self.dual_factory],
            reduced_optns)
        self.mat_vec = self.KKT_matrix.product

        # initialize the preconditioner for the KKT matrix
        self.precond = get_opt(optns, None, 'reduced', 'precond')

        if self.precond == None:
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
            self.nested = NestedKKTPreconditioner(self.KKT_matrix, embedded_krylov)
            # define preconditioner as approximate solve of KKT system
            self.precond = self.nested.product

        elif self.precond == 'idf_schur':
            # NOTE: IDF preconditioner not implemented yet
            raise NotImplementedError

        else:
            raise BadKonaOption(optns, 'reduced', 'precond')

    def _write_header(self):
        self.hist_file.write(
            '# Kona equality-constrained RSNK convergence history file\n' + \
            '# iters' + ' '*5 + \
            '   cost' + ' '*5 + \
            'optimality  ' + ' '*5 + \
            'feasibility ' + ' '*5 + \
            'objective   ' + ' '*5 + \
            'mu param    ' + '\n'
        )

    def _write_history(self, opt, feas, obj, mu):
        self.hist_file.write(
            '%7i'%self.iter + ' '*5 + \
            '%7i'%self.primal_factory._memory.cost + ' '*5 + \
            '%11e'%opt + ' '*5 + \
            '%11e'%feas + ' '*5 + \
            '%11e'%obj + ' '*5 + \
            '%11e'%mu + '\n'
        )

    def _generate_KKT_vector(self):
        primal = self.primal_factory.generate()
        dual = self.dual_factory.generate()
        return ReducedKKTVector(primal, dual)

    def solve(self):

        self._write_header()
        self.info_file.write(
            '**************************************************\n' + \
            '***        Using FLECS-based Algorithm         ***\n' + \
            '**************************************************\n')

        # generate composite KKT vectors
        X = self._generate_KKT_vector()
        P = self._generate_KKT_vector()
        P_corr = self._generate_KKT_vector()
        dLdX = self._generate_KKT_vector()
        dLdX_save = self._generate_KKT_vector()

        # generate primal vectors
        P_norm = self.primal_factory.generate()
        P_tang = self.primal_factory.generate()
        init_design = self.primal_factory.generate()
        primal_work = []
        for i in xrange(3):
            primal_work.append(self.primal_factory.generate())

        # generate state vectors
        state = self.state_factory.generate()
        state_save = self.state_factory.generate()
        adjoint = self.state_factory.generate()
        adjoint_res = self.state_factory.generate()
        state_work = []
        for i in xrange(5):
            state_work.append(self.state_factory.generate())

        # generate dual vectors
        dual_work = self.dual_factory.generate()

        # initialize basic data for outer iterations
        converged = False
        mu = self.mu_init
        self.iter = 0

        # evaluate the initial design before starting outer iterations
        X.equals_init_guess()
        init_design.equals(X._primal)
        state.equals_primal_solution(init_design)
        adjoint.equals_adjoint_solution(init_design, state, state_work[0])
        current_solution(init_design, state, adjoint, X._dual, self.iter)

        # BEGIN BIG LOOP HERE
        ###############################
        for i in xrange(self.max_iter):
            # advance iteration counter
            self.iter += 1

            self.info_file.write(
                '\n' + \
                '==========================================\n' + \
                'Beginning Major Iteration %i\n'%self.iter + \
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
                    'grad_norm0 = %e\n'%grad_norm0 + \
                    'feas_norm0 = %e\n'%feas_norm0
                )
                # calculate convergence tolerances
                grad_tol = self.primal_tol
                feas_tol = self.ceq_tol * max(feas_norm0, 1e-8)
                mu = self.mu_init
            else:
                # calculate current norms
                grad_norm = dLdX._primal.norm2
                feas_norm = dLdX._dual.norm2
                kkt_norm = sqrt(feas_norm**2 + grad_norm**2)
                # update the augmented Lagrangian penalty
                mu = max(mu, self.mu_init)*((feas_norm/feas_norm0)**self.mu_pow)
                self.info_file.write(
                    'grad_norm = %e (%e <-- tolerance)\n'%(grad_norm, grad_tol) + \
                    'feas_norm = %e (%e <-- tolerance)\n'%(feas_norm, feas_tol)
                )
            # save the current dL/dX
            dLdX_save.equals(dLdX)

            # write convergence history
            obj_val = objective_value(X._primal, state)
            self._write_history(grad_norm, feas_norm, obj_val, mu)

            # check for convergence
            if (grad_norm < grad_tol) and (feas_norm < feas_tol):
                converged = True
                break

            # compute krylov tolerances in order to achieve superlinear
            # convergence but to avoid oversolving
            krylov_tol = self.krylov.rel_tol*min(1.0, sqrt(kkt_norm/kkt_norm0))
            krylov_tol = max(krylov_tol, min(grad_tol/grad_norm, feas_tol/feas_norm))
            krylov_tol *= self.nu
            self.info_file.write('krylov tol = %e\n'%krylov_tol)

            # set ReducedKKTMatrix product tolerances
            if self.KKT_matrix.dynamic_tol:
                raise NotImplementedError('EqualityConstrainedRSNK.solve()' + \
                    'not yet set up for dynamic tolerance in product')
            else:
                self.KKT_matrix.product_fac *= krylov_tol/self.krylov.max_iter

            # set other solver and product options
            self.KKT_matrix.lamb =0.0
            self.krylov.rel_tol = krylov_tol
            self.krylov.radius = self.radius
            self.krylov.mu = mu

            self.krylov.out_file.write(
                '#-------------------------------------------------\n' + \
                '# primal solve\n')

            # reset the primal-dual step vector
            P.equals(0.0)

            # move dL/dX to right hand side
            dLdX.times(-1)

            # linearize the KKT matrix
            self.KKT_matrix.linearize(X, state, adjoint)

            # trigger the krylov solution
            self.krylov.solve(self.mat_vec, dLdX, P, self.precond)

            # START FILTER LOOP
            ######################
            # filter_success = False
            # max_filter_iter = 3
            # for j in xrange(max_filter_iter):
            #     # save old design and state before updating
            #     primal_work[0].equals(X._primal)
            #     state_save.equals(state)
            #     # update design
            #     X._primal.plus(P._primal)
            #     if state.equals_primal_solution(X._primal):
            #         # state equation solution was successful so try the filter
            #         obj = objective_value(X._primal, state)
            #         dual_work.equals_constraints(X._primal, state)
            #         cnstr_norm = dual_work.norm2
            #         if self.filter.dominates(obj, cnstr_norm):
            #             if (j == 0) and self.krylov.trust_active:
            #                 self.radius = min(2.*self.radius, self.max_radius)
            #             filter_success = True
            #             break
            #
            #     if (j == 0):
            #         # try a second order correction
            #         self.info_file.write('attempting a second-order correction...')
            #         P.equals(0.0)
            #         self.krylov.rel_tol = krylov_tol
            #         self.krylov.mu = self.mu_init
            #         self.grad_tol = 0.9*grad_norm
            #         self.feas_tol = 0.9*feas_norm
            #         self.krylov.out_file.write(
            #             '#-------------------------------------------------\n' + \
            #             '# Second-order correction (iter = %i)\n'%self.iter)
            #         dual_work.times(-1)
            #         self.krylov.apply_correction(dual_work, P)
            #         X._primal.plus(P._primal)
            #         if state.equals_primal_solution(X._primal):
            #             # state equation solution was successful so try the filter
            #             obj = objective_value(X._primal, state)
            #             dual_work.equals_constraints(X._primal, state)
            #             cnstr_norm = dual_work.norm2
            #             if self.filter.dominates(obj, cnstr_norm):
            #                 filter_success = True
            #                 self.info_file.write('successful\n')
            #                 break
            #             self.info_file.write('unsuccessful\n')
            #
            #     # if we get here, filter dominated the point
            #     # reset and shrink radius
            #     X._primal.equals(primal_work[0])
            #     state.equals(state_save)
            #     self.radius *= 0.25
            #     if (j == max_filter_iter-1):
            #         break
            #
            #     # resolve with reduced radius
            #     self.krylov.radius = self.radius
            #     self.krylov.rel_tol = krylov_tol
            #     self.krylov.mu = mu
            #     self.krylov.re_solve(dLdX, P)

            ###########################
            # END FILTER LOOP

            # if filter succeeded, then update multipliers
            # if filter_success:
            #     X._dual.plus(P._dual)

            # recalculate adjoint
            X.plus(P)
            state.equals_primal_solution(X._primal)
            adjoint.equals_adjoint_solution(X._primal, state, state_work[0])

            # write current solution
            current_solution(X._primal, state, adjoint, X._dual, self.iter)
            self.info_file.write('Norm of multipliers = %e\n'%X._dual.norm2)

        ############################
        # END OF BIG LOOP

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n'%self.iter)
