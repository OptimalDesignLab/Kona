from kona.algorithms.base_algorithm import OptimizationAlgorithm

class STCG_RSNK(OptimizationAlgorithm):
    """
    A reduced-space Newton-Krylov optimization algorithm for PDE-governed
    unconstrained problems.

    This algorithm uses a novel 2nd order adjoint formulation of the KKT
    matrix-vector product, in conjunction with a novel Krylov-method called
    FLECS for non-convex saddle point problems.

    The step produced by FLECS is globalized using a trust region approach.

    .. note::

        Insert inexact-Hessian paper reference here.

    Parameters
    ----------
    primal_factory : VectorFactory
    state_factory : VectorFactory
    dual_factory : VectorFactory, optional
    optns : dict, optional
    """
    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(STCG_RSNK, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns)

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(7)
        self.state_factory.request_num_vectors(8)

        # get other options
        self.globalization = get_opt(self.optns, 'trust', 'globalization')
        self.radius = get_opt(self.optns, 1.0, 'trust', 'init_radius')
        self.max_radius = get_opt(self.optns, 1.0, 'trust', 'max_radius')
        self.factor_matrices = get_opt(self.optns, False, 'matrix_explicit')

        # set the krylov solver
        krylov_optns = {
            'krylov_file'   : get_opt(
                self.optns, 'kona_krylov.dat', 'rsnk', 'krylov_file'),
            'subspace_size' : get_opt(self.optns, 10, 'rsnk', 'subspace_size'),
            'check_res'     : get_opt(self.optns, True, 'rsnk', 'check_res'),
            'rel_tol'       : get_opt(self.optns, 1e-2, 'rsnk', 'rel_tol'),
        }

        if self.globalization is None:
            self.info_file.write(
                ">> WARNING: Globalization is turned off! <<\n")
            self.krylov = FGMRES(self.primal_factory, krylov_optns)
        else:
            self.krylov = STCG(self.primal_factory, krylov_optns)
            self.krylov.radius = self.radius
            if self.globalization == 'linesearch':
                try:
                    line_search_alg = get_opt(
                        self.optns, StrongWolfe, 'linesearch', 'type')
                    line_search_opt = get_opt(self.optns, {}, 'linesearch')
                    self.line_search = line_search_alg(
                        line_search_opt, out_file=self.info_file)
                except Exception:
                    raise BadKonaOption(self.optns, 'linesearch', 'type')
                self.merit_func = ObjectiveMerit(
                    primal_factory, state_factory,
                    {}, self.info_file)
            elif self.globalization != 'trust':
                raise BadKonaOption(self.optns, 'globalization')

        # initialize the ReducedHessian approximation
        reduced_optns = get_opt(self.optns, {}, 'rsnk')
        reduced_optns['out_file'] = self.info_file
        self.hessian = ReducedHessian(
            [self.primal_factory, self.state_factory], reduced_optns)

        # set the Krylov solver into the Hessian object
        self.hessian.set_krylov_solver(self.krylov)

        # initialize the preconditioner to the ReducedHessian
        self.precond = get_opt(self.optns, None, 'rsnk', 'precond')
        if self.precond == 'quasi_newton':
            # set the type of quasi-Newton method
            try:
                quasi_newton = get_opt(
                    self.optns, LimitedMemoryBFGS, 'quasi_newton', 'type')
                qn_optns = get_opt(self.optns, {}, 'quasi_newton')
                qn_optns['out_file'] = self.info_file
                self.quasi_newton = quasi_newton(self.primal_factory, qn_optns)
            except Exception:
                raise BadKonaOption(self.optns, 'quasi_newton','type')
            self.precond = self.quasi_newton.solve
            self.hessian.quasi_newton = self.quasi_newton
        else:
            self.eye = IdentityMatrix()
            self.precond = self.eye.product

    def _write_header(self):
        self.hist_file.write(
            '# Kona trust-region RSNK convergence history file\n' +
            '# iters' + ' '*5 +
            '      cost' + ' '*5 +
            ' grad norm' + ' '*5 +
            ' objective' + ' '*5 +
            '     ratio' + ' '*5 +
            '    radius' + '\n'
        )

    def _write_history(self, num_iter, norm, obj, rho):
        self.hist_file.write(
            '%7i'%num_iter + ' '*5 +
            '%10i'%self.primal_factory._memory.cost + ' '*5 +
            '%10e'%norm + ' '*5 +
            '%10e'%obj + ' '*5 +
            '%10e'%rho + ' '*5 +
            '%10e'%self.radius + '\n'
        )

    def solve(self):

        x = self.primal_factory.generate()
        p = self.primal_factory.generate()
        dJdX = self.primal_factory.generate()
        dJdX_old = self.primal_factory.generate()
        primal_work = self.primal_factory.generate()
        state = self.state_factory.generate()
        adjoint = self.state_factory.generate()
        state_work = self.state_factory.generate()

        # set initial design and solve for state
        x.equals_init_design()
        if not state.equals_primal_solution(x):
            raise RuntimeError('Invalid initial point! State-solve failed.')
        # solve for adjoint
        adjoint.equals_objective_adjoint(x, state, state_work)
        # get objective value
        obj = objective_value(x, state)

        # START THE NEWTON LOOP
        #######################
        self._write_header()
        self.iter = 0
        rho = 0.0
        converged = False
        for i in xrange(self.max_iter):

            self.info_file.write(
                '==================================================\n')
            self.info_file.write('Beginning Trust-Region iteration %i\n'%(i+1))
            self.info_file.write('\n')

            # update quasi-newton
            dJdX.equals_total_gradient(x, state, adjoint)
            if i == 0:
                grad_norm0 = dJdX.norm2
                grad_norm = grad_norm0
                self.info_file.write('grad_norm0 = %e\n'%grad_norm0)
                grad_tol = self.primal_tol*grad_norm0
                dJdX_old.equals(dJdX)
            else:
                grad_norm = dJdX.norm2
                self.info_file.write(
                    'grad norm : grad_tol = %e : %e\n'%(grad_norm, grad_tol))
                dJdX_old.minus(dJdX)
                dJdX_old.times(-1.0)
                if self.precond == 'quasi_newton':
                    self.quasi_newton.add_correction(p, dJdX_old)
                dJdX_old.equals(dJdX)
            # write history
            solver_info = current_solution(
                num_iter=self.iter, curr_primal=x,
                curr_state=state, curr_adj=adjoint)
            if isinstance(solver_info, str):
                self.info_file.write('\n' + solver_info + '\n')
            self._write_history(self.iter, grad_norm, obj, rho)
            # check convergence
            if grad_norm < grad_tol:
                converged = True
                break

            # define adaptive Krylov tolerance for superlinear convergence
            krylov_tol = self.krylov.rel_tol*min(
                1.0, np.sqrt(grad_norm/grad_norm0))
            krylov_tol = max(krylov_tol, grad_tol/grad_norm)
            krylov_tol *= self.hessian.nu

            self.info_file.write('krylov tol = %e\n'%krylov_tol)
            if self.hessian.dynamic_tol:
                self.hessian.product_fac *= krylov_tol/self.krylov.max_iter

            # inexactly solve the trust-region problem
            p.equals(0.0)
            dJdX.times(-1.0)
            self.krylov.rel_tol = krylov_tol
            self.krylov.radius = self.radius
            self.hessian.linearize(x, state, adjoint)
            pred, active = self.krylov.solve(
                self.hessian.product, dJdX, p, self.precond)
            dJdX.times(-1.0)
            primal_work.equals(x)
            x.plus(p)
            x.enforce_bounds()

            if self.globalization is None:
                state.equals_primal_solution(x)
                obj = objective_value(x, state)
                adjoint.equals_objective_adjoint(x, state, state_work)
            elif self.globalization == 'trust':
                # compute the actual reduction and trust parameter rho
                obj_old = obj
                state_work.equals(state)
                if state.equals_primal_solution(x):
                    obj = objective_value(x, state)
                    rho = (obj_old - obj)/pred
                else:
                    rho = np.nan

                # update radius if necessary
                if rho < 0.1 or np.isnan(rho):
                    self.info_file.write('reverting solution...\n')
                    obj = obj_old
                    x.equals(primal_work)
                    state.equals(state_work)
                    self.radius *= 0.25
                    self.info_file.write('new radius = %f\n'%self.radius)
                else:
                    adjoint.equals_objective_adjoint(x, state, state_work)
                    if active and rho > 0.75:
                        self.radius = min(2*self.radius, self.max_radius)
                        self.info_file.write('new radius = %f\n'%self.radius)
            elif self.globalization == 'linesearch':
                # perform line search along the new direction
                p_dot_djdx = p.inner(dJdX)
                self.merit_func.reset(p, x, state, p_dot_djdx)
                alpha, _ = self.line_search.find_step_length(self.merit_func)
                # apply the step onto the primal space
                x.equals_ax_p_by(1.0, x, alpha, p)
                # adjust trust radius
                if alpha == self.line_search.alpha_max and active:
                    self.radius = min(2*self.radius, self.max_radius)
                    self.info_file.write('new radius = %f\n'%self.radius)
                elif alpha <= 0.25:
                    self.radius *= 0.25
                    self.info_file.write('new radius = %f\n'%self.radius)
            else:
                raise TypeError("Wrong globalization type!")

            if self.factor_matrices and self.iter < self.max_iter:
                factor_linear_system(x, state)

            self.iter += 1
            self.info_file.write('\n')
        #####################
        # END THE NEWTON LOOP

        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Failed to converge!\n')

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n'%self.iter)

# imports here to prevent circular errors
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, objective_value, factor_linear_system
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import LimitedMemoryBFGS, ReducedHessian
from kona.linalg.solvers.krylov import STCG, FGMRES
from kona.algorithms.util.linesearch import StrongWolfe
from kona.algorithms.util.merit import ObjectiveMerit