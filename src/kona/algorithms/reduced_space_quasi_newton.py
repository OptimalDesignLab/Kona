import sys

from kona.linalg.vectors.common import current_solution
from kona.linalg.matrices.lbfgs import LimitedMemoryBFGS
from kona.linalg.matrices.lsr1 import LimitedMemorySR1
from kona.algorithms.util.linesearch import StrongWolfe, BackTracking
from kona.algorithms.util.merit import ObjectiveMerit
from kona.options import BadKonaOption, get_opt
from kona.linalg.matrices.common import dRdU

class ReducedSpaceQuasiNewton(object):
    """
    Unconstrained optimization using quasi-Newton in the reduced space.
    """

    def __init__(self, primal_factory, state_factory, optns={},
                 out_file=sys.stdout):
        self.primal_factory = primal_factory
        self.state_factory = state_factory

        # number of vectors required in solve() method
        primal_factory.request_num_vectors(6)
        state_factory.request_num_vectors(3)

        self.info_file = get_opt(optns, sys.stdout, 'info_file')
        if isinstance(self.info_file, str):
            self.info_file = open(self.info_file,'w')

        self.max_iter = get_opt(optns, 100, 'max_iter')
        self.primal_tol = get_opt(optns, 1e-8,'primal_tol')

        # set the type of quasi-Newton method
        try:
            quasi_newton_mat = get_opt(optns, LimitedMemoryBFGS, 'quasi_newton', 'type')
            quas_newton_opts = get_opt(optns, {}, 'quasi_newton')
            self.quasi_newton = quasi_newton_mat(primal_factory, quas_newton_opts, out_file)
        except Exception as err:
            raise BadKonaOption(optns, 'quasi_newton','type')

        # set the type of line-search algorithm
        try:
            line_search_alg = get_opt(optns, StrongWolfe, 'line_search', 'type')
            line_search_opt = get_opt(optns, {}, 'line_search')
            self.line_search = line_search_alg(line_search_opt, out_file)
        except:
            raise BadKonaOption(optns, 'line_search', 'type')

        # define the merit function (which is always the objective itself here)
        merit_optns = get_opt(optns,{},'merit')
        self.merit = ObjectiveMerit(primal_factory, state_factory, merit_optns, out_file)
        self.line_search.merit_function = self.merit

    def solve(self):
        # need some way of choosing file to output to
        info = self.info_file
        # need to open the history file

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

        x.equals_init_design()
        initial_design.equals(x)
        # call current_solution

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
                self.quasi_newton.norm_init = grad_norm0
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
                self.quasi_newton.add_correction(p, dfdx_old)
                dfdx_old.equals(dfdx)

            # write convergence history here
            self.quasi_newton.solve(dfdx, p)
            p.times(-1.0)

            # set-up merit function and perform line search
            p_dot_dfdx = p.inner(dfdx)
            self.merit.reset(p, x, state, p_dot_dfdx)
            self.line_search.merit_function = self.merit
            alpha, _ = self.line_search.find_step_length()
            x.equals_ax_p_by(1.0, x, alpha, p)

            # s = delta x = alpha * p is needed later by quasi-Newton method
            p.times(alpha)

            nonlinear_sum += 1

            current_solution(x, num_iter=nonlinear_sum)

        info.write('Total number of nonlinear iterations: %i\n'%nonlinear_sum)
