import sys

from kona.options import BadKonaOption, get_opt

from kona.linalg import current_solution, factor_linear_system
from kona.linalg.matrices.hessian import LimitedMemoryBFGS

from kona.algorithms.base_algorithm import OptimizationAlgorithm
from kona.algorithms.util.linesearch import StrongWolfe
from kona.algorithms.util.merit import ObjectiveMerit

class ReducedSpaceQuasiNewton(OptimizationAlgorithm):
    """
    Unconstrained optimization using quasi-Newton in the reduced space.

    Attributes
    ----------
    approx_hessian : QuasiNewtonApprox
        Abstract matrix object that defines the QN approximation of the Hessian.
    line_search : LineSearch
        Line search object for globalization.
    """
    def __init__(self, primal_factory, state_factory=None, optns={}):
        # trigger base class initialization
        super(ReducedSpaceQuasiNewton, self).__init__(
            primal_factory, state_factory, None, optns
        )
        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(6)
        self.state_factory.request_num_vectors(3)

        # check if this problem is matrix-explicit
        self.factor_matrices = get_opt(optns, False, 'matrix_explicit')

        # set the type of quasi-Newton method
        try:
            approx_hessian = get_opt(optns, LimitedMemoryBFGS, 'quasi_newton', 'type')
            hessian_optns = get_opt(optns, {}, 'quasi_newton')
            hessian_optns['out_file'] = self.info_file
            self.approx_hessian = approx_hessian(self.primal_factory, hessian_optns)
        except Exception as err:
            raise BadKonaOption(optns, 'quasi_newton','type')

        # set up the merit function
        merit_optns = get_opt(optns,{},'merit_function')
        merit_type = get_opt(merit_optns, ObjectiveMerit, 'type')
        try:
            self.merit_func = merit_type(
                primal_factory, state_factory, merit_optns, self.info_file)
        except:
            raise BadKonaOption(optns, 'merit_function', 'type')

        # set the type of line-search algorithm
        try:
            line_search_alg = get_opt(optns, StrongWolfe, 'line_search', 'type')
            line_search_opt = get_opt(optns, {}, 'line_search')
            self.line_search = line_search_alg(line_search_opt, self.info_file)
        except:
            raise BadKonaOption(optns, 'line_search', 'type')

    def _write_header(self):
        self.hist_file.write(
            '# Kona reduced-space quasi-Newton convergence history file\n' + \
            '# iters' + ' '*5 + \
            '      cost' + ' '*5 + \
            ' grad norm' + '\n'
        )

    def _write_history(self, num_iter, norm):
        self.hist_file.write(
            '# %5i'%num_iter + ' '*5 + \
            '%10e'%self.primal_factory._memory.cost + ' '*5 + \
            '%10e'%norm + '\n'
        )

    def solve(self):
        info = self.info_file
        # get memory
        x = self.primal_factory.generate()
        p = self.primal_factory.generate()
        dfdx = self.primal_factory.generate()
        dfdx_old = self.primal_factory.generate()
        state = self.state_factory.generate()
        adjoint = self.state_factory.generate()
        state_work = self.state_factory.generate()
        initial_design = self.primal_factory.generate()
        design_work = self.primal_factory.generate()
        # initialize values into some vectors
        x.equals_init_design()
        initial_design.equals(x)
        # start optimization outer iterations
        self.iter = 0
        converged = False
        self._write_header()
        for i in xrange(self.max_iter):
            info.write('========== Outer Iteration %i ==========\n'%(i+1))
            if not state.equals_primal_solution(x):
                info.write('WARNING: Nonlinear solution failed to converge!\n')
            if self.factor_matrices:
                factor_linear_system(x, state)
            adjoint.equals_adjoint_solution(x, state, state_work)
            dfdx.equals_total_gradient(x, state, adjoint, design_work)
            # check for convergence
            if i == 0:
                grad_norm0 = dfdx.norm2
                grad_norm = grad_norm0
                self.approx_hessian.norm_init = grad_norm0
                grad_tol = self.primal_tol * grad_norm0
                info.write('grad_norm = %e : grad_tol = %e\n'%(grad_norm0, grad_tol))
                # save gradient for quasi-Newton
                dfdx_old.equals(dfdx)
            else:
                grad_norm = dfdx.norm2
                info.write('grad_norm = %e : grad_tol = %e\n'%(grad_norm, grad_tol))
                if grad_norm < grad_tol:
                    converged = True
                    break
                # update the quasi-Newton method
                dfdx_old.minus(dfdx)
                dfdx_old.times(-1.0)
                self.approx_hessian.add_correction(p, dfdx_old)
                dfdx_old.equals(dfdx)
            # write convergence history here
            current_solution(x, state, num_iter=self.iter)
            info.write('\n')
            self._write_history(self.iter, grad_norm)
            # solve for search direction
            self.approx_hessian.solve(dfdx, p)
            p.times(-1.0)
            # perform line search along the new direction
            p_dot_dfdx = p.inner(dfdx)
            self.merit_func.reset(p, x, state, p_dot_dfdx)
            alpha, _ = self.line_search.find_step_length(self.merit_func)
            # apply the step onto the primal space
            x.equals_ax_p_by(1.0, x, alpha, p)
            # s = delta x = alpha * p is needed later by quasi-Newton method
            p.times(alpha)
            self.iter += 1
        # optimization is finished, so print total number of iterations
        info.write('Total number of nonlinear iterations: %i\n'%self.iter)
