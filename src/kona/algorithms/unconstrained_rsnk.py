from kona.algorithms.base_algorithm import OptimizationAlgorithm


class UnconstrainedRSNK(OptimizationAlgorithm):
    """
    A reduced-space Newton-Krylov optimization algorithm for PDE-governed
    unconstrained problems.

    This algorithm uses a 2nd order adjoint formulation to compute 
    matrix-vector products with the Reduced Hessian.

    The product is then used in a Krylov solver to compute a Newton step.

    The step can be globalized using either line-search or trust-region 
    methods. The Krylov solver changes based on the type of globalization 
    selected by the user. Unglobalized problems are solved via FGMRES, 
    while trust-region and line-search methods use Conjugate-Gradient.

    .. note::

        Insert inexact-Hessian paper reference here.

    Parameters
    ----------
    primal_factory : VectorFactory
    state_factory : VectorFactory
    optns : dict, optional
    """

    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(UnconstrainedRSNK, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns)

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(7)
        self.state_factory.request_num_vectors(8)

        # misc attributes
        self.iter = 0

        # set the krylov solver options
        krylov_optns = {
            'krylov_file':get_opt(
                self.optns, 'kona_krylov.dat', 'rsnk', 'krylov_file'),
            'subspace_size':get_opt(self.optns, 10, 'rsnk', 'subspace_size'),
            'check_res':get_opt(self.optns, True, 'rsnk', 'check_res'),
            'rel_tol':get_opt(self.optns, 1e-2, 'rsnk', 'rel_tol'),
        }

        # determine if the underlying PDE is matrix-explicit
        self.factor_matrices = get_opt(self.optns, False, 'matrix_explicit')

        # set globalization type
        self.globalization = get_opt(self.optns, 'trust', 'globalization')
        if self.globalization is None:
            self.info_file.write(
                ">> WARNING: Globalization is turned off! <<\n")
            self.krylov = FGMRES(self.primal_factory, krylov_optns)
        elif self.globalization == 'linesearch':
            self.krylov = LineSearchCG(self.primal_factory, krylov_optns)
            line_search_opt = get_opt(self.optns, {}, 'linesearch')
            self.line_search = BackTracking(
                line_search_opt, out_file=self.info_file)
            self.merit_func = ObjectiveMerit(
                primal_factory, state_factory,
                {}, self.info_file)
            self.last_alpha = 1.0
        elif self.globalization == 'trust':
            self.krylov = STCG(self.primal_factory, krylov_optns)
            self.radius = get_opt(self.optns, 1.0, 'trust', 'init_radius')
            self.max_radius = get_opt(self.optns, 1.0, 'trust', 'max_radius')
            self.krylov.radius = self.radius
        else:
            raise BadKonaOption(self.optns, 'globalization')

            # initialize the ReducedHessian approximation
        reduced_optns = get_opt(self.optns, {}, 'rsnk')
        reduced_optns['out_file'] = self.info_file
        self.hessian = ReducedHessian(
            [self.primal_factory, self.state_factory], reduced_optns)

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
                raise BadKonaOption(self.optns, 'quasi_newton', 'type')
            self.precond = self.quasi_newton.solve
            self.hessian.quasi_newton = self.quasi_newton
        else:
            self.eye = IdentityMatrix()
            self.precond = self.eye.product

    def _write_header(self, obj_scale):
        if self.globalization == 'trust':
            glob_text = 'radius    '
        elif self.globalization == 'linesearch':
            glob_text = 'alpha     '
        else:
            glob_text = '          '
        self.hist_file.write(
            '# Kona %s RSNK convergence history file '%self.globalization +
            '(grad scale = %e)\n'%obj_scale +
            '# iters' +
            '      cost' + ' '*5 +
            'grad norm ' + ' '*7 +
            'objective ' + ' '*7 +
            glob_text + '\n'
        )

    def _write_history(self, num_iter, norm, obj):
        if self.globalization == 'trust':
            glob_num = '%f'%self.radius
        elif self.globalization == 'linesearch':
            glob_num = '%f'%self.last_alpha
        else:
            glob_num = ''
        self.hist_file.write(
            '%7i'%num_iter +
            '%10i'%self.primal_factory._memory.cost + ' '*5 +
            '%8e'%norm + ' '*5 +
            '%8e'%obj + ' '*5 +
            glob_num + '\n'
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
        # get objective scale
        dJdX_old.equals_total_gradient(x, state, adjoint)
        grad_norm0 = dJdX_old.norm2
        obj_scale = 1.
        # recompute adjoint and grad norm
        adjoint.equals_objective_adjoint(x, state, state_work, scale=obj_scale)
        dJdX_old.equals_total_gradient(x, state, adjoint, scale=obj_scale)

        # START THE NEWTON LOOP
        #######################
        self._write_header(obj_scale)
        converged = False
        grad_tol = self.primal_tol*grad_norm0*obj_scale
        for i in xrange(self.max_iter):

            self.info_file.write(
                '==================================================\n')
            self.info_file.write('Beginning Newton iteration %i\n'%(i + 1))
            self.info_file.write('\n')

            # compute convergence norm
            dJdX.equals_total_gradient(x, state, adjoint, scale=obj_scale)
            grad_norm = dJdX.norm2
            self.info_file.write(
                'grad norm : grad_tol = %e : %e\n'%(grad_norm, grad_tol))

            # update quasi-newton preconditioner
            if self.precond == 'quasi_newton' and i != 0:
                dJdX_old.minus(dJdX)
                dJdX_old.times(-1.0)
                self.quasi_newton.add_correction(p, dJdX_old)
            dJdX_old.equals(dJdX)

            # write history 
            solver_info = current_solution(
                num_iter=self.iter, curr_primal=x,
                curr_state=state, curr_adj=adjoint)
            if isinstance(solver_info, str):
                self.info_file.write('\n' + solver_info + '\n')
            obj = objective_value(x, state)
            self._write_history(self.iter, grad_norm, obj)
            # check convergence
            if grad_norm < grad_tol:
                converged = True
                break

            # define adaptive Krylov tolerance for superlinear convergence
            krylov_tol = self.krylov.rel_tol*min(
                1.0, np.sqrt(grad_norm/grad_norm0))
            krylov_tol = max(krylov_tol, grad_tol/grad_norm)
            self.krylov.rel_tol = krylov_tol
            self.info_file.write('krylov tol = %e\n'%krylov_tol)

            # inexactly solve the trust-region problem
            p.equals(0.0)
            dJdX.times(-1.0)
            self.hessian.linearize(x, state, adjoint, scale=obj_scale)
            pred, active = self.krylov.solve(
                self.hessian.product, dJdX, p, self.precond)
            dJdX.times(-1.0)

            if p.norm2 == 0.0:
                # we converged to a local minimum that may or may not satisfy tolerances
                self.info_file.write("\nNewton-step is zero!\n")
                break

            if self.globalization is None:
                x.plus(p)
                x.enforce_bounds()
                state.equals_primal_solution(x)
                if self.factor_matrices:
                    factor_linear_system(x, state)
                adjoint.equals_objective_adjoint(x, state, state_work, scale=obj_scale)

            elif self.globalization == 'trust':
                # store old design point and take the step
                primal_work.equals(x)
                x.plus(p)
                x.enforce_bounds()

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
                    x.equals(primal_work)
                    state.equals(state_work)
                    self.radius *= 0.25
                    self.krylov.radius = self.radius
                    self.info_file.write('new radius = %f\n'%self.radius)
                else:
                    if self.factor_matrices:
                        factor_linear_system(x, state)
                    adjoint.equals_objective_adjoint(x, state, state_work, scale=obj_scale)
                    if active and rho > 0.75:
                        self.radius = min(2*self.radius, self.max_radius)
                        self.krylov.radius = self.radius
                        self.info_file.write('new radius = %f\n'%self.radius)

            elif self.globalization == 'linesearch':
                # perform line search along the new direction
                p_dot_djdx = p.inner(dJdX)
                self.info_file.write('p_dot_djdx = %e\n'%p_dot_djdx)
                self.merit_func.reset(p, x, state, p_dot_djdx)
                alpha, _ = self.line_search.find_step_length(self.merit_func)
                self.last_alpha = alpha
                # apply the step onto the primal space
                p.times(alpha)
                x.plus(p)
                state.equals_primal_solution(x)
                if self.factor_matrices:
                    factor_linear_system(x, state)
                adjoint.equals_objective_adjoint(x, state, state_work, scale=obj_scale)
            else:
                raise TypeError("Wrong globalization type!")

            self.iter += 1
            self.info_file.write('\n')
        #####################
        # END THE NEWTON LOOP

        if converged:
            self.info_file.write('\nOptimization successful!\n')
        else:
            self.info_file.write('\nFailed to converge!\n')

        self.info_file.write(
            '\nTotal number of nonlinear iterations: %i\n'%self.iter)


# imports here to prevent circular errors
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, objective_value, factor_linear_system
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import LimitedMemoryBFGS, ReducedHessian
from kona.linalg.solvers.krylov import STCG, LineSearchCG, FGMRES
from kona.algorithms.util.linesearch import BackTracking
from kona.algorithms.util.merit import ObjectiveMerit
