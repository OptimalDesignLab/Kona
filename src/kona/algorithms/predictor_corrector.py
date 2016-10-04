from kona.algorithms.base_algorithm import OptimizationAlgorithm

class PredictorCorrector(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory,
                 eq_factory=None, ineq_factory=None, optns={}):
        # trigger base class initialization
        super(PredictorCorrector, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(12)
        self.state_factory.request_num_vectors(5)

        # general options
        ############################################################
        self.factor_matrices = get_opt(optns, False, 'matrix_explicit')

        # reduced hessian settings
        ############################################################
        self.hessian = ReducedHessian(
            [self.primal_factory, self.state_factory])
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
        self.krylov = FGMRES(self.primal_factory, krylov_optns)

        # homotopy options
        ############################################################
        self.lamb = 0.0
        self.inner_tol = get_opt(optns, 1e-2, 'homotopy', 'inner_tol')
        self.inner_maxiter = get_opt(optns, 50, 'homotopy', 'inner_maxiter')
        self.step = get_opt(
            optns, 0.05, 'homotopy', 'init_step')
        self.nom_dcurve = get_opt(optns, 1.0, 'homotopy', 'nominal_dist')
        self.nom_angl = get_opt(
            optns, 5.0*np.pi/180., 'homotopy', 'nominal_angle')
        self.max_factor = get_opt(optns, 2.0, 'homotopy', 'max_factor')
        self.min_factor = get_opt(optns, 0.5, 'homotopy', 'min_factor')

    def _write_header(self, tol):
        self.hist_file.write(
            '# Kona Param. Contn. convergence history (opt tol = %e)\n'%tol +
            '# outer' + ' '*5 +
            '  inner' + ' '*5 +
            '   cost' + ' '*5 +
            'objective   ' + ' ' * 5 +
            'opt grad    ' + ' '*5 +
            'homotopy    ' + ' ' * 5 +
            'hom grad    ' + ' '*5 +
            'lambda      ' + ' '*5 +
            '\n'
        )

    def _write_outer(self, outer, obj, opt_norm):
        self.hist_file.write(
            '%7i' % outer + ' ' * 5 +
            '-'*7 + ' ' * 5 +
            '%7i' % self.primal_factory._memory.cost + ' ' * 5 +
            '%11e' % obj + ' ' * 5 +
            '%11e' % opt_norm + ' ' * 5 +
            '-'*11 + ' ' * 5 +
            '-'*11 + ' ' * 5 +
            '%11e' % self.lamb + ' ' * 5 +
            '\n'
        )

    def _write_inner(self, outer, inner, obj, opt_norm, hom, hom_opt):
        self.hist_file.write(
            '%7i' % outer + ' ' * 5 +
            '%7i' % inner + ' ' * 5 +
            '%7i' % self.primal_factory._memory.cost + ' ' * 5 +
            '%11e' % obj + ' ' * 5 +
            '%11e' % opt_norm + ' ' * 5 +
            '%11e' % hom + ' ' * 5 +
            '%11e' % hom_opt + ' ' * 5 +
            '%11e' % self.lamb + ' ' * 5 +
            '\n'
        )

    def _mat_vec(self, in_vec, out_vec):
        self.hessian.product(in_vec, out_vec)
        out_vec.times(self.lamb)
        self.prod_work.equals(in_vec)
        self.prod_work.times(1. - self.lamb)
        out_vec.plus(self.prod_work)

    def solve(self):
        self.info_file.write(
            '\n' +
            '**************************************************\n' +
            '***        Using Parameter Continuation        ***\n' +
            '**************************************************\n' +
            '\n')

        # get the vectors we need
        x = self.primal_factory.generate()
        x0 = self.primal_factory.generate()
        x_save = self.primal_factory.generate()
        dJdX = self.primal_factory.generate()
        dJdX_hom = self.primal_factory.generate()
        primal_work = self.primal_factory.generate()
        dx = self.primal_factory.generate()
        dx_newt = self.primal_factory.generate()
        rhs_vec = self.primal_factory.generate()
        t = self.primal_factory.generate()
        t_save = self.primal_factory.generate()
        self.prod_work = self.primal_factory.generate()

        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        state_save = self.state_factory.generate()
        adj = self.state_factory.generate()
        adj_save = self.state_factory.generate()

        # initialize the problem at the starting point
        x0.equals_init_design()
        x.equals(x0)
        if not state.equals_primal_solution(x):
            raise RuntimeError('Invalid initial point! State-solve failed.')
        if self.factor_matrices:
            factor_linear_system(x, state)

        # solve for objective adjoint
        adj.equals_objective_adjoint(x, state, state_work)

        # compute initial gradient
        dJdX.equals_total_gradient(x, state, adj)
        grad_norm0 = dJdX.norm2
        grad_fac = 1. / grad_norm0
        dJdX.times(grad_fac)
        grad_tol = grad_norm0 * grad_fac * self.primal_tol
        self._write_header(grad_tol)

        # write the initial conditions
        obj0 = objective_value(x, state)
        self._write_outer(0, obj0, grad_norm0)

        # set up the predictor RHS
        rhs_vec.equals(dJdX)
        primal_work.equals(x)
        primal_work.minus(x0)
        primal_work.times(-1.)
        rhs_vec.plus(primal_work)
        rhs_vec.times(-1.)

        # compute initial tangent vector
        adj.times(grad_fac)
        self.hessian.linearize(x, state, adj, scale=grad_fac)
        t.equals(0.0)
        self.krylov.solve(self._mat_vec, rhs_vec, t, self.precond)

        # normalize tangent vector
        tnorm = np.sqrt(t.inner(t) + 1.0)
        t.times(1. / tnorm)
        dlamb = 1. / tnorm

        # START PREDICTOR ITERATIONS
        ############################
        outer_iters = 1
        total_iters = 0
        while self.lamb < 1.0 and outer_iters <= self.max_iter:

            self.info_file.write(
                '==================================================\n')
            self.info_file.write(
                'Outer Homotopy iteration %i\n'%(outer_iters+1))
            self.info_file.write('\n')

            # compute optimality metrics
            grad_norm = dJdX.norm2
            self.info_file.write(
                'grad_norm : grad_tol = %e : %e\n'%(grad_norm, self.primal_tol))

            # save current solution in case we need to revert
            x_save.equals(x)
            state_save.equals(state)
            adj_save.equals(adj)
            t_save.equals(t)
            dlamb_save = dlamb
            lamb_save = self.lamb

            # take a predictor step
            x.equals_ax_p_by(1.0, x, self.step, t)
            self.lamb += dlamb * self.step
            if self.lamb > 1.0:
                self.lamb = 1.0
            self.info_file.write('\nlamb          = %f\n' % self.lamb)

            # solve states
            if not state.equals_primal_solution(x):
                raise RuntimeError(
                    'Invalid predictor point! State-solve failed.')
            if self.factor_matrices:
                factor_linear_system(x, state)

            # solve for objective adjoint
            adj.equals_objective_adjoint(x, state, state_work, scale=grad_fac)

            # START CORRECTOR (Newton) ITERATIONS
            #####################################
            max_newton = self.inner_maxiter
            inner_iters = 0
            dx_newt.equals(0.0)
            for i in xrange(max_newton):

                self.info_file.write('\n')
                self.info_file.write('   Inner Newton iteration %i\n'%(i+1))
                self.info_file.write('   -------------------------------\n')

                # compute the homotopy map derivative
                dJdX.equals_total_gradient(
                    x, state, adj, scale=grad_fac)
                dJdX_hom.equals(dJdX)
                dJdX_hom.times(self.lamb)
                primal_work.equals(x)
                primal_work.minus(x0)
                xTx = primal_work.inner(primal_work)
                primal_work.times((1. - self.lamb))
                dJdX_hom.plus(primal_work)

                # get convergence norms
                if i == 0:
                    newt_norm0 = dJdX_hom.norm2
                    newt_tol = self.inner_tol * newt_norm0
                    if newt_tol < self.primal_tol or self.lamb == 1.0:
                        newt_tol = self.primal_tol
                newt_norm = dJdX_hom.norm2
                self.info_file.write(
                    '   newt_norm : newt_tol = %e : %e\n'%(
                        newt_norm, newt_tol))

                # compute homotopy map value and write history
                obj = objective_value(x, state)
                hom = self.lamb*obj/grad_norm0
                hom += 0.5*(1 - self.lamb)*xTx
                grad_norm = dJdX.norm2
                self._write_inner(
                    outer_iters, inner_iters,
                    obj, grad_norm,
                    hom, newt_norm)

                # send solution to solver
                solver_info = current_solution(
                    num_iter=total_iters + 1, curr_primal=x,
                    curr_state=state, curr_adj=adj)
                if isinstance(solver_info, str) and solver_info != '':
                    self.info_file.write('\n' + solver_info + '\n')

                # check convergence
                if newt_norm <= newt_tol:
                    self.info_file.write('\n   Corrector step converged!\n')
                    break

                # linearize the hessian at the new point
                self.hessian.linearize(x, state, adj, scale=grad_fac)

                # define the RHS vector for the homotopy system
                dJdX_hom.times(-1.)

                # solve the system
                dx.equals(0.0)
                self.krylov.solve(self._mat_vec, dJdX_hom, dx, self.precond)
                dx_newt.plus(dx)

                # update the design
                x.plus(dx)
                if not state.equals_primal_solution(x):
                    raise RuntimeError('Newton step failed!')
                if self.factor_matrices:
                    factor_linear_system(x, state)
                adj.equals_objective_adjoint(
                    x, state, state_work, scale=grad_fac)

                # advance iter counter
                inner_iters += 1
                total_iters += 1

            # if we finished the corrector step at mu=1, we're done!
            if self.lamb == 1.:
                self.info_file.write('\n>> Optimization DONE! <<\n')
                return

            # COMPUTE NEW TANGENT VECTOR
            ############################

            # set up the predictor RHS
            rhs_vec.equals(dJdX)
            primal_work.equals(x)
            primal_work.minus(x0)
            primal_work.times(-1.)
            rhs_vec.plus(primal_work)
            rhs_vec.times(-1.)

            # solve for the new tangent vector
            self.hessian.linearize(x, state, adj, scale=grad_fac)
            t.equals(0.0)
            self.krylov.solve(self._mat_vec, rhs_vec, t, self.precond)

            # normalize tangent vector
            tnorm = np.sqrt(t.inner(t) + 1.0)
            t.times(1. / tnorm)
            dlamb = 1. / tnorm

            # compute distance to curve and step angles
            self.info_file.write('\n')
            dcurve = dx_newt.norm2
            self.info_file.write(
                'dist to curve = %e\n'%dcurve)

            # compute angle
            angl = np.arccos(t.inner(t_save) + (dlamb * dlamb_save))
            self.info_file.write(
                'angle         = %f\n'%(angl*180./np.pi))

            # compute deceleration factor
            dfac = max(np.sqrt(dcurve/self.nom_dcurve), angl/self.nom_angl)
            dfac = max(min(dfac, self.max_factor), self.min_factor)

            # apply the deceleration factor
            self.info_file.write('factor        = %f\n' % dfac)
            self.step /= dfac
            self.info_file.write('step len      = %f\n' % self.step)

            # if deceleration factor hit the upper limit
            if dfac == self.max_factor:
                self.info_file.write(
                    'High curvature! Reverting solution...\n')
                # revert solution
                x.equals(x_save)
                self.lamb = lamb_save
                t.equals(t_save)
                dlamb = dlamb_save
                state.equals(state_save)
                adj.equals(adj_save)
                if self.factor_matrices:
                    factor_linear_system(x, state)

            # update iteration counters
            outer_iters += 1
            self.hist_file.write('\n')
            self.info_file.write('\n')

# imports here to prevent circular errors
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, factor_linear_system, objective_value
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedHessian
from kona.linalg.solvers.krylov import FGMRES