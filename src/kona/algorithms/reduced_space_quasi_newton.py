import sys

from kona.linalg.matrices.lbfgs import LimitedMemoryBFGS
from kona.linalg.matrices.lsr1 import LimitedMemorySR1
from kona.algorithms.util.linesearch import StrongWolfe, BackTracking
from kona.algorithms.util.merit import ObjectiveMerit
from kona.errors import BadKonaOption
from kona.linalg.matrices.common import dRdU

class ReducedSpaceQuasiNewton(object):
    """
    Unconstrained optimization using quasi-Newton in the reduced space.
    """

    def __init__(self, primal_factory, state_factory, optns,
                 out_file=sys.stdout):
        self.primal_factory = primal_factory
        self.state_factory = state_factory

        # number of vectors required in solve() method
        primal_factory.request_num_vectors(6)
        state_factory.request_num_vectors(2)

        # set the type of quasi-Newton method
        if optns['quasi_newton']['type'] == 'lbfgs':
            self.quasi_newton = LimitedMemoryBFGS(primal_factory, optns['quasi_newton'],
                                                  out_file)
        elif optns['quasi_newton']['type'] == 'sr1':
            self.quasi_newton = LimitedMemorySR1(primal_factory, optns['quasi_newton'],
                                                 out_file)

        qn = optns['quasi_newton']['type']
        if qn:
            self.quasi_newton = qn(primal_factory, optns['quasi_newton'],out_file)
        else:
            raise BadKonaOption(optns, ('quasi_newton', 'type'))

        # set the type of line-search algorithm
        if optns['line_search']['type'] == 'wolfe':
            self.line_search = StrongWolfe(optns['line_search'], out_file)
        elif optns['line_search']['type'] == 'back_track':
            self.line_search = BackTracking(optns['line_search'], out_file)
        else:
            raise BadKonaOption(optns, ('line_search', 'type'))

        # define the merit function (which is always the objective itself here)
        self.merit = ObjectiveMerit(optns['merit'], primal_factory, out_file)
        self.line_search.set_merit_function(self.merit)

    def solve():
        # need some way of choosing file to output to
        info = open(optns['info_file'], 'w')
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

        x.equals_initial_design()
        initial_design.equals(x)
        # call current_solution

        nonlinear_sum = 0
        converged = False
        for i in xrange(optns['max_iter']):
            state.equals_primal_solution(x)
            adjoint.equals_adjoint_solution(x, state, state_work)
            dfdx.equals_reduced_gradient(x, state, adjoint, design_work)
            # check for convergence
            if i == 0:
                grad_norm0 = dfdx.norm2
                grad_norm = grad_norm0
                self.quasi_newton.norm_init = grad_norm0
                info.write('grad_norm0 = ', grad_norm0, '\n')
                grad_tol = optns['primal_tol'] * grad_norm0
                # save gradient for quasi-Newton
                dfdx_old.equals(dfdx)
            else:
                grad_norm = dfdx.norm2
                info.write('grad_norm/grad_norm0 = ',grad_norm/grad_norm0,'\n')
                if grad_norm < grad_tol:
                    converged = True
                    break
                # update the quasi-Newton method
                dfdx_old.minus(dfdx)
                dfdx_old.times(-1.0)
                self.quasi_newton.add_correction(p, dfdx)
                dfdx_old.equals(dfdx)

            # write convergence history here
            self.quasi_newton.solve(dfdx, p)
            p.times(-1.0)

            # set-up merit function and perform line search
            self.merit.reset(p, x, p_dot_grad, state, adjoint)
            self.line_search.set_search_dot_grad(p_dot_grad)
            alpha = self.line_search.find_step_length()
            x.equals_ax_plus_by(1.0, x, alpha, p)

            adjoint.equals_adjoint_solution(x, state)

            # s = delta x = alpha * p is needed later by quasi-Newton method
            p.times(alpha)

            nonlinear_sum += 1

        info.write('Total number of nonlinear iterations:',
                   nonlinear_sum, '\n')
