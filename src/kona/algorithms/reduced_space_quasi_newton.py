from kona.algorithms.base_algorithm import OptimizationAlgorithm

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
    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(ReducedSpaceQuasiNewton, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns)
        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(6)
        self.state_factory.request_num_vectors(3)

        # check if this problem is matrix-explicit
        self.factor_matrices = get_opt(self.optns, False, 'matrix_explicit')

        # set the type of quasi-Newton method
        try:
            approx_hessian = get_opt(
                self.optns, LimitedMemoryBFGS, 'quasi_newton', 'type')
            hessian_optns = get_opt(self.optns, {}, 'quasi_newton')
            hessian_optns['out_file'] = self.info_file
            self.approx_hessian = approx_hessian(
                self.primal_factory, hessian_optns)
        except Exception:
            raise BadKonaOption(self.optns, 'quasi_newton','type')

        # set the type of line-search algorithm and merit function
        self.globalization = get_opt(self.optns, 'linesearch', 'globalization')
        if self.globalization is not None:
            try:
                line_search_alg = get_opt(
                    self.optns, StrongWolfe, 'lineseach', 'type')
                line_search_opt = get_opt(self.optns, {}, 'linesearch')
                self.line_search = line_search_alg(
                    line_search_opt, self.info_file)
            except Exception:
                raise BadKonaOption(self.optns, 'linesearch', 'type')
            merit_type = get_opt(
                self.optns, ObjectiveMerit, 'merit_function', 'type')
            if merit_type is ObjectiveMerit:
                try:
                    merit_opt = get_opt(self.optns, {}, 'merit_function')
                    self.merit_func = merit_type(
                        primal_factory, state_factory,
                        merit_opt, self.info_file)
                except Exception:
                    raise BadKonaOption(self.optns, 'merit_function')
            else:
                raise TypeError('Invalid merit function!')

    def _write_header(self):
        self.hist_file.write(
            '# Kona reduced-space quasi-Newton convergence history file\n' +
            '# iters' + ' '*5 +
            '      cost' + ' '*5 +
            ' grad norm' + '\n'
        )

    def _write_history(self, num_iter, norm):
        self.hist_file.write(
            '%7i'%num_iter + ' '*5 +
            '%10i'%self.primal_factory._memory.cost + ' '*5 +
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
            adjoint.equals_objective_adjoint(x, state, state_work)
            dfdx.equals_total_gradient(x, state, adjoint)
            # check for convergence
            if i == 0:
                grad_norm0 = dfdx.norm2
                grad_norm = grad_norm0
                self.approx_hessian.norm_init = grad_norm0
                grad_tol = self.primal_tol * grad_norm0
                info.write(
                    'grad_norm = %e : grad_tol = %e\n'%(grad_norm0, grad_tol))
                # save gradient for quasi-Newton
                dfdx_old.equals(dfdx)
            else:
                grad_norm = dfdx.norm2
                info.write(
                    'grad_norm = %e : grad_tol = %e\n'%(grad_norm, grad_tol))
                if grad_norm < grad_tol:
                    converged = True
                    break
                # update the quasi-Newton method
                dfdx_old.minus(dfdx)
                dfdx_old.times(-1.0)
                self.approx_hessian.add_correction(p, dfdx_old)
                dfdx_old.equals(dfdx)
            # write convergence history here
            solver_info = current_solution(
                num_iter=self.iter, curr_primal=x, curr_state=state)
            if isinstance(solver_info, str):
                info.write('\n' + solver_info + '\n')
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
        solver_info = current_solution(
            num_iter=self.iter, curr_primal=x, curr_state=state)
        if isinstance(solver_info, str):
            info.write('\n' + solver_info + '\n')
        info.write('\n')
        self._write_history(self.iter, grad_norm)
        if converged:
            self.info_file.write('Optimization successful!\n')
        else:
            self.info_file.write('Failed to converge!\n')

        info.write('Total number of nonlinear iterations: %i\n'%self.iter)

# imports here to prevent circular errors
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, factor_linear_system
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.algorithms.util.linesearch import StrongWolfe
from kona.algorithms.util.merit import ObjectiveMerit