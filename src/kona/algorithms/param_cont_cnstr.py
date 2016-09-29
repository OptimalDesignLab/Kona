from kona.algorithms.base_algorithm import OptimizationAlgorithm

class ParameterContCnstr(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory,
                 eq_factory=None, ineq_factory=None, optns={}):
        # trigger base class initialization
        super(ParameterContCnstr, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(11)
        self.state_factory.request_num_vectors(5)
        self.eq_factory.request_num_vectors(10)

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
        self.mu = get_opt(optns, 0.0, 'homotopy', 'lambda')
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
            ' '*7 + ' ' * 5 +
            '%7i' % self.primal_factory._memory.cost + ' ' * 5 +
            '%11e' % obj + ' ' * 5 +
            '%11e' % opt_norm + ' ' * 5 +
            '%11e' % feas_norm + ' ' * 5 +
            ' '*11 + ' ' * 5 +
            ' '*11 + ' ' * 5 +
            ' '*11 + ' ' * 5 +
            '%11e' % self.mu + ' ' * 5 +
            '\n'
        )

    def _write_inner(self, outer, inner,
                       hom, hom_opt, hom_feas):
        self.hist_file.write(
            '%7i' % outer + ' ' * 5 +
            '%7i' % inner + ' ' * 5 +
            '%7i' % self.primal_factory._memory.cost + ' ' * 5 +
            ' '*11 + ' ' * 5 +
            ' '*11 + ' ' * 5 +
            ' '*11 + ' ' * 5 +
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
        prod_work = self._generate_kkt()

        primal_work = self._generate_primal()

        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        state_save = self.state_factory.generate()
        adj = self.state_factory.generate()
        adj_save = self.state_factory.generate()
        adj_obj = self.state_factory.generate()
        adj_cnstr = self.state_factory.generate()

        c0 = self._generate_dual()
        dual_work = self._generate_dual()

        # initialize the problem at the starting point
        x0.equals_init_guess()
        x.equals(x0)
        if not state.equals_primal_solution(x):
            raise RuntimeError('Invalid initial point! State-solve failed.')
        c0.equals_constraints(x0, state)
        if self.factor_matrices:
            factor_linear_system(x, state)

        # compute scaling factors
        adj_obj.equals_objective_adjoint(x.primal, state, state_work)
        primal_work.equals_total_gradient(x.primal, state, adj_obj)
        dual_work.equals_constraints(x.primal, state)
        obj_norm0 = primal_work.norm2
        cnstr_norm0 = dual_work.norm2
        self._write_header(self.primal_tol, self.cnstr_tol)

        # define a hessian product that includes the homotopy term
        def mat_vec(in_vec, out_vec):
            prod_work.equals(in_vec)
            self.hessian.product(in_vec, out_vec)
            prod_work.primal.times(1. - self.mu)
            out_vec.primal.plus(prod_work)

        # START PREDICTOR ITERATIONS
        ############################
        outer_iters = 0
        total_iters = 0
        while self.mu < 1.0 and outer_iters < self.max_iter:

            self.info_file.write(
                '==================================================\n')
            self.info_file.write(
                'Outer Homotopy iteration %i\n'%(outer_iters+1))
            self.info_file.write('\n')

            # compute KKT conditions
            adj_obj.equals_objective_adjoint(
                x.primal, state, state_work, scale=1./obj_norm0)
            adj_cnstr.equals_constraint_adjoint(
                x.primal, state, x.dual, state_work, scale=1./cnstr_norm0)
            adj.equals_ax_p_by(1., adj_obj, 1., adj_cnstr)
            dJdX.equals_KKT_conditions(
                x, state, adj,
                obj_scale=1./obj_norm0, cnstr_scale=1./cnstr_norm0)

            # compute convergence parameters
            if outer_iters == 0:
                opt_norm0 = dJdX.primal.norm2
                feas_norm0 = dJdX.dual.norm2
                opt_tol = opt_norm0 * self.primal_tol
                feas_tol = feas_norm0 * self.cnstr_tol
                opt_norm = opt_norm0
                feas_norm = feas_norm0
                self._write_header(opt_tol, feas_tol)
            else:
                opt_norm = dJdX.primal.norm2
                feas_norm = dJdX.dual.norm2
            self.info_file.write(
                'opt_norm : opt_tol = %e : %e\n'%(
                    opt_norm, opt_tol) +
                'feas_norm : feas_tol = %e : %e\n' % (
                    feas_norm, feas_tol))

            # write outer history
            obj = objective_value(x.primal, state)
            self._write_outer(outer_iters + 1, obj, opt_norm, feas_norm)

            # check convergence
            if opt_norm <= opt_tol and feas_norm <= feas_tol:
                break

            # initialize dfac, dx and dmu with values
            if outer_iters == 0:
                t.equals(0.0)
                dmu = 1.
                dfac = 1.0

            # apply the deceleration factor
            self.info_file.write('factor        = %f\n'%dfac)
            self.step /= dfac
            self.info_file.write('step len      = %f\n'%self.step)

            # save dx and dmu from previous iteration
            t_save.equals(t)
            dmu_save = dmu

            # compute predictor RHS
            adj_obj.times(self.mu)
            rhs_vec.primal.equals_total_gradient(x.primal, state, adj_obj,
                                                 scale=self.mu / obj_norm0)
            primal_work.equals(x)
            primal_work.minus(x0)
            primal_work.times(-1.)
            rhs_vec.primal.plus(primal_work)
            rhs_vec.dual.equals(c0)
            rhs_vec.dual.times(1./cnstr_norm0)
            rhs_vec.times(-1.)

            # compute homotopy adjoint
            adj.equals_ax_p_by(1., adj_obj, 1., adj_cnstr)

            # solve for the predictor step (tangent vector)
            t.equals(0.0)
            self.hessian.linearize(
                x, state, adj,
                obj_scale=self.mu/obj_norm0, cnstr_scale=1./cnstr_norm0)
            self.krylov.solve(mat_vec, rhs_vec, t, self.precond)

            # normalize the tangent vector
            tnorm = np.sqrt(t.inner(t) + 1.)
            t.times(1./tnorm)
            dmu = 1./tnorm

            # update lambda
            self.info_file.write('d_lambda      = %f\n'%(dmu * self.step))
            mu_save = self.mu
            self.mu += dmu * self.step
            if self.mu > 1.0:
                self.mu = 1.0
            self.info_file.write('lambda        = %f\n' % self.mu)

            # take the predictor step
            x_save.equals(x)
            x.equals_ax_p_by(1.0, x, self.step, t)

            # update state
            state_save.equals(state)
            if not state.equals_primal_solution(x.primal):
                raise RuntimeError('Predictor step failed!')
            if self.factor_matrices:
                factor_linear_system(x.primal, state)

            # save the adjoint in case we need to revert
            adj_save.equals(adj)

            # START CORRECTOR (Newton) ITERATIONS
            #####################################
            max_newton = self.inner_maxiter
            inner_iters = 0
            for i in xrange(max_newton):

                self.info_file.write('\n')
                self.info_file.write('   Inner Newton iteration %i\n'%(i+1))
                self.info_file.write('   -------------------------------\n')

                # save solution
                solver_info = current_solution(
                    num_iter=total_iters + 1, curr_primal=x.primal,
                    curr_state=state, curr_adj=adj, curr_dual=x.dual)
                if isinstance(solver_info, str):
                    self.info_file.write('\n' + solver_info + '\n')

                # compute the homotopy map derivatives
                adj.equals_lagrangian_adjoint(
                    x, state, state_work,
                    obj_scale=self.mu/obj_norm0, cnstr_scale=1./cnstr_norm0)
                dJdX_hom.equals_KKT_conditions(
                    x, state, adj,
                    obj_scale=self.mu/obj_norm0, cnstr_scale=1./cnstr_norm0)
                primal_work.equals(x.primal)
                primal_work.minus(x0)
                xTx = primal_work.norm2**2
                primal_work.times(1. - self.mu)
                dJdX_hom.primal.plus(primal_work)
                dual_work.equals(c0)
                dual_work.times(1. - self.mu)
                dJdX.dual.minus(dual_work)

                # get convergence norms
                if i == 0:
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
                hom = self.mu * obj / obj_norm0
                hom += 0.5*(1 - self.mu) * xTx
                self._write_inner(
                    outer_iters+1, inner_iters+1,
                    hom, hom_opt_norm, hom_feas_norm)

                # check convergence
                if hom_opt_norm <= hom_opt_tol and hom_feas_norm <= hom_feas_tol:
                    self.info_file.write('   Corrector step converged!\n')
                    break

                # advance iter counter
                inner_iters += 1
                total_iters += 1

                # linearize the hessian at the new point
                self.hessian.linearize(
                    x, state, adj,
                    obj_scale=self.mu/obj_norm0, cnstr_scale=1./cnstr_norm0)

                # define the RHS vector for the homotopy system
                dJdX_hom.times(-1.)

                # solve the system
                dx.equals(0.0)
                self.krylov.solve(mat_vec, dJdX_hom, dx, self.precond)
                dx_newt.plus(dx)

                # update the design
                x.plus(dx)
                if not state.equals_primal_solution(x):
                    raise RuntimeError('Newton step failed!')
                if self.factor_matrices:
                    factor_linear_system(x, state)

            # compute distance to curve and step angles
            self.info_file.write('\n')
            dcurve = dx_newt.norm2
            self.info_file.write(
                'dist to curve = %e\n'%dcurve)

            # compute angle
            if outer_iters == 0:
                angl = 0.0
            else:
                angl = np.arccos(
                    t.inner(t_save) + (dmu * dmu_save))
            self.info_file.write(
                'angle         = %f\n'%(angl*180./np.pi))

            # compute deceleration factor
            dfac = max(np.sqrt(dcurve/self.nom_dcurve), angl/self.nom_angl)
            dfac = max(min(dfac, self.max_factor), self.min_factor)

            # if deceleration factor hit the upper limit
            if dfac == self.max_factor and self.mu < 1.0:
                self.info_file.write(
                    'High curvature! Reverting solution...\n')
                # revert solution
                x.equals(x_save)
                self.mu = mu_save
                t.equals(t_save)
                dmu = dmu_save
                state.equals(state_save)
                if self.factor_matrices:
                    factor_linear_system(x.primal, state)

            # update iteration counters
            outer_iters += 1
            self.hist_file.write('\n')
            self.info_file.write('\n')

# imports here to prevent circular errors
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.solvers.krylov import FGMRES