import numpy as np

from kona.options import BadKonaOption, get_opt

from kona.linalg import current_solution, objective_value, factor_linear_system
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import LimitedMemoryBFGS, ReducedHessian
from kona.linalg.solvers.krylov import STCG

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
                 dual_factory=None, optns={}):
        # trigger base class initialization
        super(STCG_RSNK, self).__init__(
            primal_factory, state_factory, dual_factory, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(7)
        self.state_factory.request_num_vectors(8)

        # get other options
        self.globalization = get_opt(optns, 'trust', 'globalization')
        self.radius = get_opt(optns, 1.0, 'trust', 'init_radius')
        self.max_radius = get_opt(optns, 1.0, 'trust', 'max_radius')
        self.factor_matrices = get_opt(optns, False, 'matrix_explicit')

        # set the krylov solver
        krylov_optns = {
            'krylov_file'   : get_opt(
                optns, 'kona_krylov.dat', 'rsnk', 'krylov_file'),
            'subspace_size' : get_opt(optns, 10, 'rsnk', 'subspace_size'),
            'check_res'     : get_opt(optns, True, 'rsnk', 'check_res'),
            'rel_tol'       : get_opt(optns, 1e-2, 'rsnk', 'rel_tol'),
        }
        self.krylov = STCG(self.primal_factory, krylov_optns)
        self.krylov.radius = self.radius

        # initialize the ReducedHessian approximation
        reduced_optns = get_opt(optns, {}, 'rsnk')
        reduced_optns['out_file'] = self.info_file
        self.hessian = ReducedHessian(
            [self.primal_factory, self.state_factory], reduced_optns)

        # set the Krylov solver into the Hessian object
        self.hessian.set_krylov_solver(self.krylov)

        # initialize the preconditioner to the ReducedHessian
        self.precond = get_opt(optns, None, 'rsnk', 'precond')
        if self.precond == 'quasi_newton':
            # set the type of quasi-Newton method
            try:
                quasi_newton = get_opt(
                    optns, LimitedMemoryBFGS, 'quasi_newton', 'type')
                qn_optns = get_opt(optns, {}, 'quasi_newton')
                qn_optns['out_file'] = self.info_file
                self.quasi_newton = quasi_newton(self.primal_factory, qn_optns)
            except Exception:
                raise BadKonaOption(optns, 'quasi_newton','type')
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
        adjoint.equals_adjoint_solution(x, state, state_work)
        # get objective value
        obj = objective_value(x, state)

        # START THE NEWTON LOOP
        #######################
        self._write_header()
        self.iter = 0
        rho = 0.0
        for i in xrange(self.max_iter):

            self.info_file.write(
                '==================================================\n')
            self.info_file.write('Beginning Trust-Region iteration %i\n'%(i+1))
            self.info_file.write('\n')

            # update quasi-newton
            dJdX.equals_total_gradient(x, state, adjoint, primal_work)
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
            current_solution(x, state, adjoint, num_iter=self.iter)
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
                adjoint.equals_adjoint_solution(x, state, state_work)
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
                if rho < 0.25 or np.isnan(rho):
                    self.radius *= 0.25
                else:
                    if active and (rho > 0.75):
                        self.radius = min(2*self.radius, self.max_radius)

                # revert the solution if necessary
                if rho < 0.1:
                    self.info_file.write('reverting solution...\n')
                    obj = obj_old
                    x.equals(primal_work)
                    state.equals(state_work)
                else:
                    adjoint.equals_adjoint_solution(x, state, state_work)
            else:
                raise TypeError("Wrong globalization type!")

            if self.factor_matrices and self.iter < self.max_iter:
                factor_linear_system(x, state)

            self.iter += 1
        #####################
        # END THE NEWTON LOOP

        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Failed to converge!\n')

        self.info_file.write(
            'Total number of nonlinear iterations: %i\n'%self.iter)
