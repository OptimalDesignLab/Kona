from kona.algorithms.base_algorithm import OptimizationAlgorithm

class ParameterContinuation(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory,
                 eq_factory=None, ineq_factory=None, optns={}):
        # trigger base class initialization
        super(ParameterContinuation, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(11)
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
        self.lamb = get_opt(optns, 0.0, 'homotopy', 'lambda')
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
            'opt grad    ' + ' '*5 +
            'objective   ' + ' '*5 +
            'hom grad    ' + ' '*5 +
            'homotopy    ' + ' '*5 +
            'lambda      ' + ' '*5 +
            '\n'
        )

    def _write_history(self, outer, inner, opt_grad, obj, hom_grad, hom):
        self.hist_file.write(
            '%7i'%outer + ' '*5 +
            '%7i'%inner + ' '*5 +
            '%7i'%self.primal_factory._memory.cost + ' '*5 +
            '%11e'%opt_grad + ' '*5 +
            '%11e'%obj + ' '*5 +
            '%11e'%hom_grad + ' '*5 +
            '%11e'%hom + ' '*5 +
            '%11e'%self.lamb + ' '*5 +
            '\n'
        )

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
        adj.equals_adjoint_solution(x, state, state_work)

        # compute initial gradient
        dJdX.equals_total_gradient(x, state, adj)
        grad_norm0 = dJdX.norm2
        self._write_header(self.primal_tol)

        # define a homotopy mat-vec at the current lambda
        def mat_vec(in_vec, out_vec):
            self.hessian.product(in_vec, out_vec)
            out_vec.times(self.lamb/grad_norm0)
            primal_work.equals(in_vec)
            primal_work.times(1. - self.lamb)
            out_vec.plus(primal_work)

        # START PREDICTOR ITERATIONS
        ############################
        outer_iters = 0
        total_iters = 0
        while self.lamb < 1.0 and outer_iters < self.max_iter:

            self.info_file.write(
                '==================================================\n')
            self.info_file.write(
                'Outer Homotopy iteration %i\n'%(outer_iters+1))
            self.info_file.write('\n')

            # compute optimality metrics
            dJdX.equals_total_gradient(x, state, adj)
            dJdX.times(1./grad_norm0)
            grad_norm = dJdX.norm2
            self.info_file.write(
                'grad_norm : grad_tol = %e : %e\n'%(grad_norm, self.primal_tol))

            # check convergence
            if grad_norm < self.primal_tol:
                break

            # compute the predictor RHS
            rhs_vec.equals(dJdX)
            primal_work.equals(x)
            primal_work.minus(x0)
            primal_work.times(-1.)
            rhs_vec.plus(primal_work)
            rhs_vec.times(-1.)

            # initialize dfac, dx and dlamb with values
            if outer_iters == 0:
                t.equals(0.0)
                dlamb = 1.
                dfac = 1.0

            # apply the deceleration factor
            self.info_file.write('factor        = %f\n'%dfac)
            self.step /= dfac
            self.info_file.write('step len      = %f\n'%self.step)

            # save dx and dlamb from previous iteration
            t_save.equals(t)
            dlamb_save = dlamb

            # solve for the predictor step (tangent vector)
            t.equals(0.0)
            self.hessian.linearize(x, state, adj)
            self.krylov.solve(mat_vec, rhs_vec, t, self.precond)

            # normalize the tangent vector
            tnorm = np.sqrt(t.inner(t) + 1.)
            t.times(1./tnorm)
            dlamb = 1./tnorm

            # update lambda
            self.info_file.write('d_lambda      = %f\n'%(dlamb * self.step))
            lamb_save = self.lamb
            self.lamb += dlamb * self.step
            if self.lamb > 1.0:
                self.lamb = 1.0
            self.info_file.write('lambda        = %f\n'%self.lamb)

            # take the predictor step
            x_save.equals(x)
            x.equals_ax_p_by(1.0, x, self.step, t)

            # update state and adjoint
            state_save.equals(state)
            if not state.equals_primal_solution(x):
                raise RuntimeError('Predictor step failed!')
            if self.factor_matrices:
                factor_linear_system(x, state)
            adj_save.equals(adj)
            adj.equals_adjoint_solution(x, state, state_work)

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
                    num_iter=total_iters + 1, curr_primal=x,
                    curr_state=state, curr_adj=adj)
                if isinstance(solver_info, str):
                    self.info_file.write('\n' + solver_info + '\n')

                # compute the homotopy map derivative
                dJdX_hom.equals_total_gradient(x, state, adj)
                dJdX_hom.times(1./grad_norm0)
                opt_grad = dJdX_hom.norm2
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
                    if newt_tol < self.primal_tol:
                        newt_tol = self.primal_tol
                newt_norm = dJdX_hom.norm2
                self.info_file.write(
                    '   newt_norm : newt_tol = %e : %e\n'%(
                        newt_norm, newt_tol))

                obj = objective_value(x, state)
                hom = self.lamb*obj/grad_norm0
                hom += 0.5*(1 - self.lamb)*xTx
                self._write_history(
                    outer_iters+1, inner_iters+1,
                    opt_grad, obj,
                    dJdX_hom.norm2, hom)

                # check convergence
                if newt_norm < newt_tol:
                    self.info_file.write('   Corrector step converged!\n')
                    break

                # advance iter counter
                inner_iters += 1
                total_iters += 1

                # linearize the hessian at the new point
                self.hessian.linearize(x, state, adj)

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
                adj.equals_adjoint_solution(x, state, state_work)

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
                    t.inner(t_save) + (dlamb * dlamb_save))
            self.info_file.write(
                'angle         = %f\n'%(angl*180./np.pi))

            # compute deceleration factor
            dfac = max(np.sqrt(dcurve/self.nom_dcurve), angl/self.nom_angl)
            dfac = max(min(dfac, self.max_factor), self.min_factor)

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
                if self.factor_matrices:
                    factor_linear_system(x, state)
                adj.equals(adj_save)

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