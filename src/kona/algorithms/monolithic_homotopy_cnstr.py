from kona.algorithms.base_algorithm import OptimizationAlgorithm

class MonolithicHomotopyCnstr(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory,
                 eq_factory=None, ineq_factory=None, optns={}):
        # trigger base class initialization
        super(MonolithicHomotopyCnstr, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(9)
        self.state_factory.request_num_vectors(4)
        self.eq_factory.request_num_vectors(9)

        # general options
        ############################################################
        self.factor_matrices = get_opt(optns, False, 'matrix_explicit')

        # reduced hessian settings
        ############################################################
        self.hessian = ReducedKKTMatrix(
            [self.primal_factory, self.state_factory, self.eq_factory])
        self.mat_vec = self.hessian.product

        # hessian preconditiner settings
        ############################################################
        self.precond = get_opt(optns, None, 'rsnk', 'precond')
        if self.precond is None:
            # use identity matrix product as preconditioner
            self.eye = IdentityMatrix()
            self.precond = self.eye.product
        else:
            raise BadKonaOption(optns, 'rsnk', 'precond')

        # krylov solver settings
        ############################################################
        krylov_optns = {
            'krylov_file'   : get_opt(
                optns, 'kona_krylov.dat', 'rsnk', 'krylov_file'),
            'subspace_size' : get_opt(optns, 10, 'rsnk', 'subspace_size'),
            'check_res'     : get_opt(optns, True, 'rsnk', 'check_res'),
            'rel_tol'       : get_opt(optns, 1e-2, 'rsnk', 'rel_tol'),
        }
        self.krylov = FGMRES(self.primal_factory, krylov_optns,
                             eq_factory=self.eq_factory, ineq_factory=None)

        # homotopy options
        ############################################################
        self.mu = 1.0
        self.dmu = get_opt(
            optns, 0.2, 'homotopy', 'init_step')
        self.max_dmu = get_opt(optns, self.dmu, 'homotopy', 'max_step')
        self.min_dmu = get_opt(optns, 0.25*self.dmu, 'homotopy', 'min_step')
        self.max_factor = get_opt(optns, 2.0, 'homotopy', 'max_factor')
        self.min_factor = get_opt(optns, 1./3., 'homotopy', 'min_factor')

    def _write_header(self):
        self.hist_file.write(
            '# Kona Monolithic Homotopy convergence history\n' +
            '# iters' + ' '*5 +
            '   cost' + ' '*5 +
            'objective   ' + ' ' * 5 +
            'opt norm    ' + ' '*5 +
            'feas norm   ' + ' '*5 +
            'homotopy    ' + ' ' * 5 +
            'hom opt     ' + ' '*5 +
            'hom feas    ' + ' '*5 +
            'mu          ' + ' '*5 +
            '\n'
        )

    def _write_history(self, iters,
                    obj, opt_norm, feas_norm,
                    hom, hom_opt, hom_feas):
        self.hist_file.write(
            '%7i' % iters + ' ' * 5 +
            '%7i' % self.primal_factory._memory.cost + ' ' * 5 +
            '%11e' % obj + ' ' * 5 +
            '%11e' % opt_norm + ' ' * 5 +
            '%11e' % feas_norm + ' ' * 5 +
            '%11e' % hom + ' ' * 5 +
            '%11e' % hom_opt + ' ' * 5 +
            '%11e' % hom_feas + ' ' * 5 +
            '%11e' % self.mu + ' ' * 5 +
            '\n'
        )

    def _generate_primal(self):
        return self.primal_factory.generate()

    def _generate_dual(self):
        return self.eq_factory.generate()

    def _generate_kkt(self):
        primal = self._generate_primal()
        dual = self._generate_dual()
        return ReducedKKTVector(primal, dual)

    def _mat_vec(self, in_vec, out_vec):
        self.hessian.product(in_vec, out_vec)
        out_vec.times(self.mu)
        self.prod_work.equals(in_vec)
        self.prod_work.dual.times(-1.)
        self.prod_work.times(1 - self.mu)
        out_vec.plus(self.prod_work)

    def solve(self):
        self.info_file.write(
            '\n' +
            '**************************************************\n' +
            '***        Using Parameter Continuation        ***\n' +
            '**************************************************\n' +
            '\n')

        # get the vectors we need
        x = self._generate_kkt()
        x0 = self._generate_kkt()
        dJdX = self._generate_kkt()
        hom_map = self._generate_kkt()
        dJdX_hom = self._generate_kkt()
        dx = self._generate_kkt()
        rhs_vec = self._generate_kkt()
        self.prod_work = self._generate_kkt()

        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        adj = self.state_factory.generate()
        adj_obj = self.state_factory.generate()

        primal_work = self._generate_primal()

        c0 = self._generate_dual()

        # initialize the problem at the starting point
        x0.equals_init_guess()
        x.equals(x0)
        if not state.equals_primal_solution(x.primal):
            raise RuntimeError('Invalid initial point! State-solve failed.')
        if self.factor_matrices:
            factor_linear_system(x.primal, state)

        # compute scaling factors
        adj_obj.equals_objective_adjoint(x.primal, state, state_work)
        primal_work.equals_total_gradient(x.primal, state, adj_obj)
        obj_norm0 = primal_work.norm2
        c0.equals_constraints(x.primal, state)
        cnstr_norm0 = c0.norm2
        obj_fac = 1./obj_norm0
        cnstr_fac = 1./cnstr_norm0

        # compute the lagrangian adjoint
        adj.equals_lagrangian_adjoint(
            x, state, state_work,
            obj_scale=obj_fac, cnstr_scale=cnstr_fac)

        # write header to hist
        self._write_header()

        # start MH iterations
        iters = 0
        while iters < self.max_iter:

            self.info_file.write(
                '==================================================\n')
            self.info_file.write(
                'Monolithic Homotopy iteration %i\n'%(iters+1))
            self.info_file.write('\n')

            # compute KKT conditions
            dJdX.equals_KKT_conditions(
                x, state, adj,
                obj_scale=obj_fac, cnstr_scale=cnstr_fac)

            # compute the homotopy term/map
            hom_map.primal.equals(x.primal)
            hom_map.primal.minus(x0.primal)
            xTx = hom_map.primal.norm2**2
            hom_map.dual.equals(x.dual)
            hom_map.dual.minus(x0.dual)
            mTm = hom_map.dual.norm2**2
            hom_map.dual.times(-1.)

            # assemble the complete convex homotopy
            dJdX_hom.equals_ax_p_by(1 - self.mu, dJdX, self.mu, hom_map)

            # compute convergence parameters
            opt_norm = dJdX.primal.norm2
            feas_norm = dJdX.dual.norm2
            hom_opt = dJdX_hom.primal.norm2
            hom_feas = dJdX_hom.dual.norm2
            self.info_file.write(
                'opt_norm : feas_norm = %e : %e\n'%(opt_norm, feas_norm) +
                'hom_opt  : hom_feas  = %e : %e\n'%(hom_opt, hom_feas))

            # write outer history
            obj = objective_value(x.primal, state)
            hom = (1. - self.mu) * obj_fac * obj
            hom += 0.5 * self.mu * xTx
            hom -= 0.5 * self.mu * mTm
            self._write_history(
                iters, obj, opt_norm, feas_norm, hom, hom_opt, hom_feas)

            # save solution
            solver_info = current_solution(
                num_iter=iters+1, curr_primal=x.primal,
                curr_state=state, curr_adj=adj, curr_dual=x.dual)
            if isinstance(solver_info, str) and solver_info != '':
                self.info_file.write('\n' + solver_info + '\n')

            if self.mu == 0.0:
                self.info_file.write('\n >> Optimization DONE! <<\n')
                return

            # assemble RHS for step solution
            gamma = 1./self.dmu
            rhs_vec.equals(dJdX_hom)
            rhs_vec.times(gamma)
            rhs_vec.minus(hom_map)
            rhs_vec.plus(dJdX)
            rhs_vec.times(-1.)

            # prepare the KKT matrix
            self.hessian.linearize(
                x, state, adj, obj_scale=obj_fac, cnstr_scale=cnstr_fac)

            # solve for the step
            dx.equals(0.0)
            self.info_file.write('\nSolving step with mu = %f\n'%self.mu)
            self.krylov.solve(self._mat_vec, rhs_vec, dx, self.precond)

            # step length adaptation
            if iters == 0:
                dx_targ = dx.norm2 * self.dmu
            else:
                dmu_prev = self.dmu
                self.dmu = dx.norm2 / dx_targ
                self.dmu = 1./self.dmu
                fac = self.dmu / dmu_prev
                fac = max(min(fac, self.max_factor), self.min_factor)
                self.dmu = fac * dmu_prev

            if self.dmu < self.min_dmu and self.mu > 0.0:
                self.dmu = self.min_dmu
            elif self.dmu > self.max_dmu:
                self.dmu = self.max_dmu

            mu_star = self.mu - self.dmu
            min_mu = 0.25 * self.mu

            if mu_star < 0.0:
                self.dmu = self.mu
            elif mu_star < min_mu:
                self.dmu = abs(min_mu - self.mu)

            # take step
            self.info_file.write('\nAccepting step with d_mu = %f\n'%self.dmu)
            dx.times(self.dmu)
            x.plus(dx)
            self.mu -= self.dmu

            # advance iteration count
            iters += 1
            self.info_file.write('\n')

# imports here to prevent circular errors
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.solvers.krylov import FGMRES