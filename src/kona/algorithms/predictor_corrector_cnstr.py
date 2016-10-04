from kona.algorithms.base_algorithm import OptimizationAlgorithm

class PredictorCorrectorCnstr(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory,
                 eq_factory=None, ineq_factory=None, optns={}):
        # trigger base class initialization
        super(PredictorCorrectorCnstr, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(12)
        self.state_factory.request_num_vectors(6)
        self.eq_factory.request_num_vectors(13)

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
        self.mu = 0.0
        self.inner_tol = get_opt(optns, 1e-2, 'homotopy', 'inner_tol')
        self.inner_maxiter = get_opt(optns, 50, 'homotopy', 'inner_maxiter')
        self.step = get_opt(
            optns, 0.05, 'homotopy', 'init_step')
        self.nom_dcurve = get_opt(optns, 1.0, 'homotopy', 'nominal_dist')
        self.nom_angl = get_opt(
            optns, 5.0*np.pi/180., 'homotopy', 'nominal_angle')
        self.max_factor = get_opt(optns, 2.0, 'homotopy', 'max_factor')
        self.min_factor = get_opt(optns, 0.5, 'homotopy', 'min_factor')

    def _write_header(self, opt_tol, feas_tol):
        self.hist_file.write(
            '# Kona Param. Contn. convergence history ' +
            '(opt tol = %e | feas tol = %e)\n'%(opt_tol, feas_tol) +
            '# outer' + ' '*5 +
            '  inner' + ' '*5 +
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

    def _write_outer(self, outer, obj, opt_norm, feas_norm):
        self.hist_file.write(
            '%7i' % outer + ' ' * 5 +
            '-'*7 + ' ' * 5 +
            '%7i' % self.primal_factory._memory.cost + ' ' * 5 +
            '%11e' % obj + ' ' * 5 +
            '%11e' % opt_norm + ' ' * 5 +
            '%11e' % feas_norm + ' ' * 5 +
            '-'*12 + ' ' * 5 +
            '-'*12 + ' ' * 5 +
            '-'*12 + ' ' * 5 +
            '%11e' % self.mu + ' ' * 5 +
            '\n'
        )

    def _write_inner(self, outer, inner,
                     obj, opt_norm, feas_norm,
                     hom, hom_opt, hom_feas):
        self.hist_file.write(
            '%7i' % outer + ' ' * 5 +
            '%7i' % inner + ' ' * 5 +
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
        x_save = self._generate_kkt()
        dJdX = self._generate_kkt()
        dJdX_hom = self._generate_kkt()
        dx = self._generate_kkt()
        dx_newt = self._generate_kkt()
        rhs_vec = self._generate_kkt()
        t = self._generate_kkt()
        t_save = self._generate_kkt()
        self.prod_work = self._generate_kkt()

        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        state_save = self.state_factory.generate()
        adj = self.state_factory.generate()
        adj_save = self.state_factory.generate()
        adj_obj = self.state_factory.generate()

        primal_work = self._generate_primal()

        c0 = self._generate_dual()
        dual_work = self._generate_dual()

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

        # compute initial KKT conditions
        dJdX.equals_KKT_conditions(
            x, state, adj,
            obj_scale=obj_fac, cnstr_scale=cnstr_fac)

        # compute convergence metrics
        opt_norm0 = dJdX.primal.norm2
        feas_norm0 = dJdX.dual.norm2
        opt_tol = opt_norm0 * self.primal_tol
        feas_tol = feas_norm0 * self.cnstr_tol
        self._write_header(opt_tol, feas_tol)

        # write the initial point
        obj0 = objective_value(x.primal, state)
        self._write_outer(0, obj0, opt_norm0, feas_norm0)
        self.hist_file.write('\n')

        # set up predictor RHS
        rhs_vec.equals(dJdX)
        primal_work.equals(x.primal)
        primal_work.minus(x0.primal)
        primal_work.times(-1.)
        rhs_vec.primal.plus(primal_work)
        dual_work.equals(x.dual)
        dual_work.minus(x0.dual)
        rhs_vec.dual.plus(dual_work)
        rhs_vec.times(-1.)

        # linearize the KKT matrix and solve for the initial tangent vector
        self.hessian.linearize(
            x, state, adj,
            obj_scale=obj_fac, cnstr_scale=cnstr_fac)
        t.equals(0.0)
        self.krylov.solve(self._mat_vec, rhs_vec, t, self.precond)

        # normalize tangent vector
        tnorm = np.sqrt(t.inner(t) + 1.0)
        t.times(1./tnorm)
        dmu = 1./tnorm

        # START OUTER ITERATIONS
        #########################
        outer_iters = 1
        total_iters = 0
        while self.mu < 1.0 and outer_iters <= self.max_iter:

            self.info_file.write(
                '==================================================\n')
            self.info_file.write(
                'Outer Homotopy iteration %i\n'%(outer_iters+1))
            self.info_file.write('\n')

            opt_norm = dJdX.primal.norm2
            feas_norm = dJdX.dual.norm2
            self.info_file.write(
                'opt_norm : opt_tol = %e : %e\n'%(
                    opt_norm, opt_tol) +
                'feas_norm : feas_tol = %e : %e\n' % (
                    feas_norm, feas_tol))

            # save current solution in case we need to revert
            x_save.equals(x)
            state_save.equals(state)
            adj_save.equals(adj)
            t_save.equals(t)
            dmu_save = dmu
            mu_save = self.mu

            # take a predictor step
            x.equals_ax_p_by(1.0, x, self.step, t)
            self.mu += dmu * self.step
            if self.mu > 1.0:
                self.mu = 1.0
            self.info_file.write('\nmu            = %f\n'%self.mu)

            # solve states
            if not state.equals_primal_solution(x.primal):
                raise RuntimeError(
                    'Invalid predictor point! State-solve failed.')
            if self.factor_matrices:
                factor_linear_system(x.primal, state)

            # solve adjoint
            adj.equals_lagrangian_adjoint(
                x, state, state_work,
                obj_scale=obj_fac, cnstr_scale=cnstr_fac)

            # START CORRECTOR (Newton) ITERATIONS
            #####################################
            max_newton = self.inner_maxiter
            inner_iters = 0
            dx_newt.equals(0.0)
            for i in xrange(max_newton):

                self.info_file.write('\n')
                self.info_file.write('   Inner Newton iteration %i\n'%(i+1))
                self.info_file.write('   -------------------------------\n')

                # compute the homotopy map derivatives
                dJdX.equals_KKT_conditions(
                    x, state, adj,
                    obj_scale=obj_fac, cnstr_scale=cnstr_fac)
                dJdX_hom.equals(dJdX)
                dJdX_hom.times(self.mu)
                primal_work.equals(x.primal)
                primal_work.minus(x0.primal)
                xTx = primal_work.norm2**2
                primal_work.times(1. - self.mu)
                dJdX_hom.primal.plus(primal_work)
                dual_work.equals(x.dual)
                dual_work.minus(x0.dual)
                mTm = dual_work.norm2**2
                dual_work.times(1. - self.mu)
                dJdX_hom.dual.minus(dual_work)

                # get convergence norms
                if inner_iters == 0:
                    # compute optimality norms
                    hom_opt_norm0 = dJdX_hom.primal.norm2
                    hom_opt_norm = hom_opt_norm0
                    hom_opt_tol = self.inner_tol * hom_opt_norm0
                    if hom_opt_tol < opt_tol or self.mu == 1.0:
                        hom_opt_tol = opt_tol
                    # compute feasibility norms
                    hom_feas_norm0 = dJdX_hom.dual.norm2
                    hom_feas_norm = hom_feas_norm0
                    hom_feas_tol = self.inner_tol * hom_feas_norm0
                    if hom_feas_tol < feas_tol or self.mu == 1.0:
                        hom_feas_tol = feas_tol
                else:
                    hom_opt_norm = dJdX_hom.primal.norm2
                    hom_feas_norm = dJdX_hom.dual.norm2
                self.info_file.write(
                    '   hom_opt_norm : hom_opt_tol = %e : %e\n'%(
                        hom_opt_norm, hom_opt_tol) +
                    '   hom_feas_norm : hom_feas_tol = %e : %e\n'%(
                        hom_feas_norm, hom_feas_tol))

                # write inner history
                obj = objective_value(x.primal, state)
                hom = self.mu * obj_fac * obj
                hom += 0.5*(1 - self.mu) * xTx
                hom -= 0.5*(1 - self.mu) * mTm
                opt_norm = dJdX.primal.norm2
                feas_norm = dJdX.dual.norm2
                self._write_inner(
                    outer_iters, inner_iters,
                    obj, opt_norm, feas_norm,
                    hom, hom_opt_norm, hom_feas_norm)

                # save solution
                solver_info = current_solution(
                    num_iter=total_iters + 1, curr_primal=x.primal,
                    curr_state=state, curr_adj=adj, curr_dual=x.dual)
                if isinstance(solver_info, str) and solver_info != '':
                    self.info_file.write('\n' + solver_info + '\n')

                # check convergence
                if hom_opt_norm <= hom_opt_tol and hom_feas_norm <= hom_feas_tol:
                    self.info_file.write('\n   Corrector step converged!\n')
                    break

                # linearize the hessian at the new point
                self.hessian.linearize(
                    x, state, adj,
                    obj_scale=obj_fac, cnstr_scale=cnstr_fac)

                # define the RHS vector for the homotopy system
                dJdX_hom.times(-1.)

                # solve the system
                dx.equals(0.0)
                self.krylov.solve(self._mat_vec, dJdX_hom, dx, self.precond)
                dx_newt.plus(dx)

                # update the design
                x.plus(dx)
                if not state.equals_primal_solution(x.primal):
                    raise RuntimeError('Newton step failed!')
                if self.factor_matrices:
                    factor_linear_system(x.primal, state)

                # compute the adjoint
                adj.equals_lagrangian_adjoint(
                    x, state, state_work,
                    obj_scale=obj_fac, cnstr_scale=cnstr_fac)

                # advance iter counter
                inner_iters += 1
                total_iters += 1

            # if we finished the corrector step at mu=1, we're done!
            if self.mu == 1.:
                self.info_file.write('\n>> Optimization DONE! <<\n')
                return

            # COMPUTE NEW TANGENT VECTOR
            ############################

            # assemble the predictor RHS
            rhs_vec.equals(dJdX)
            primal_work.equals(x.primal)
            primal_work.minus(x0.primal)
            primal_work.times(-1.)
            rhs_vec.primal.plus(primal_work)
            dual_work.equals(x.dual)
            dual_work.minus(x0.dual)
            rhs_vec.dual.plus(dual_work)
            rhs_vec.times(-1.)

            # compute the new tangent vector and predictor step
            t.equals(0.0)
            self.hessian.linearize(
                x, state, adj,
                obj_scale=obj_fac, cnstr_scale=cnstr_fac)
            self.krylov.solve(self._mat_vec, rhs_vec, t, self.precond)

            # normalize the tangent vector
            tnorm = np.sqrt(t.inner(t) + 1.0)
            t.times(1./tnorm)
            dmu = 1./tnorm

            # compute distance to curve
            self.info_file.write('\n')
            dcurve = dx_newt.norm2
            self.info_file.write(
                'dist to curve = %e\n' % dcurve)

            # compute angle between steps
            uTv = t.inner(t_save) + (dmu * dmu_save)
            angl = np.arccos(uTv)
            self.info_file.write(
                'angle         = %f\n' % (angl * 180. / np.pi))

            # compute deceleration factor
            dfac = max(np.sqrt(dcurve / self.nom_dcurve), angl / self.nom_angl)
            dfac = max(min(dfac, self.max_factor), self.min_factor)

            # apply the deceleration factor
            self.info_file.write('factor        = %f\n' % dfac)
            self.step /= dfac
            self.info_file.write('step len      = %f\n' % self.step)

            # if factor is bad, go back to the previous point with new factor
            if dfac == self.max_factor:
                self.info_file.write(
                    'High curvature! Rejecting solution...\n')
                # revert solution
                x.equals(x_save)
                self.mu = mu_save
                t.equals(t_save)
                dmu = dmu_save
                state.equals(state_save)
                adj.equals(adj_save)
                if self.factor_matrices:
                    factor_linear_system(x.primal, state)

            # advance iteration counter
            outer_iters += 1
            self.info_file.write('\n')
            self.hist_file.write('\n')

# imports here to prevent circular errors
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.solvers.krylov import FGMRES