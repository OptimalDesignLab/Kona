import sys

from kona.options import BadKonaOption, get_opt

from kona.linalg.vectors.common import current_solution
from kona.linalg.matrices.common import dRdU
from kona.linalg.matrices.hessian import LimitedMemoryBFGS

from kona.algorithms.base_algorithm import OptimizationAlgorithm
from kona.algorithms.util.linesearch import StrongWolfe

class ReducedSpaceQuasiNewton(OptimizationAlgorithm):
    """
    Unconstrained optimization using quasi-Newton in the reduced space.

    Attributes
    ----------
    approx_hessian : QuasiNewtonApprox-like
        Abstract matrix object that defines the QN approximation of the Hessian.
    line_search : LineSearch-like
        Line search object for globalization.
    """
    def __init__(self, primal_factory, state_factory, optns={}):
        # trigger base class initialization
        super(ReducedSpaceQuasiNewton, self).__init__(
            primal_factory, state_factory, None, optns
        )
        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(6)
        self.state_factory.request_num_vectors(3)
        # set the type of quasi-Newton method
        try:
            approx_hessian = get_opt(optns, LimitedMemoryBFGS, 'quasi_newton', 'type')
            hessian_optns = get_opt(optns, {}, 'quasi_newton')
            self.approx_hessian = approx_hessian(self.primal_factory, hessian_optns)
        except Exception as err:
            raise BadKonaOption(optns, 'quasi_newton','type')
        # set the type of line-search algorithm
        try:
            line_search_alg = get_opt(optns, StrongWolfe, 'line_search', 'type')
            line_search_opt = get_opt(optns, {}, 'line_search')
            self.line_search = line_search_alg(line_search_opt)
        except:
            raise BadKonaOption(optns, 'line_search', 'type')

    def solve(self):
        # need some way of choosing file to output to
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
        nonlinear_sum = 0
        converged = False
        for i in xrange(self.max_iter):
            info.write('========== Outer Iteration %i ==========\n'%(i+1))
            state.equals_primal_solution(x)
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
            current_solution(x, num_iter=nonlinear_sum)
            info.write('\n')
            # solve for search direction
            self.approx_hessian.solve(dfdx, p)
            p.times(-1.0)
            # perform line search along the new direction
            p_dot_dfdx = p.inner(dfdx)
            self.merit.reset(p, x, state, p_dot_dfdx)
            alpha, _ = self.line_search.find_step_length(self.merit)
            # apply the step onto the primal space
            x.equals_ax_p_by(1.0, x, alpha, p)
            # s = delta x = alpha * p is needed later by quasi-Newton method
            p.times(alpha)
            nonlinear_sum += 1
        # optimization is finished, so print total number of iterations
        info.write('Total number of nonlinear iterations: %i\n'%nonlinear_sum)
