import sys

from kona.linalg.vectors.common import objective_value
from kona.linalg.solvers.util import EPS
from kona.linalg.matrices.common import dRdX

class ObjectiveMerit(object):
    """
    Merit function for line searches applied to the raw objective

    Attributes
    ----------
    alpha_prev : float
        The value of the step size on the previous iteration
    obj_start : float
        The value of the objective at the beginning of the line search
    p_dot_grad : float
        The product :math:`\langle p, \nabla f \rangle` at the beginning
    x_start : PrimalVector
        Initial position of the variable, where :math:`\alpha = 0`
    search_dir : PrimalVector
        The search direction adopted
    state_work : StateVector
        Work vector for state
    adjoint_work : StateVector
        Work vector for adjoint
    design_work : PrimalVector
        Work vector for primal
    """

    def __init__(self, primal_factory, state_factory, optns=None, out_file=sys.stdout):
        self.primal_factory = primal_factory
        self.primal_factory.request_num_vectors(2)
        self.state_factory = state_factory
        self.state_factory.request_num_vectors(1)
        self.out_file = out_file
        self._allocated = False

    def reset(self, search_dir, x_start, u_start, p_dot_grad):
        # if the internal vectors are not allocated, do it now
        if not self._allocated:
            self.x_trial = self.primal_factory.generate()
            self.primal_work = self.primal_factory.generate()
            self.adjoint_work = self.state_factory.generate()
            self._allocated = True
        # store information for the new point the merit function is reset at
        self.search_dir = search_dir
        self.x_start = x_start
        self.u_trial = u_start
        self.func_val = objective_value(x_start, u_start)
        self.p_dot_grad = p_dot_grad
        self.last_func_alpha = 0.0
        self.last_grad_alpha = 0.0

    def eval_func(self, alpha):
        # do calculations only if alpha changed significantly
        if abs(alpha - self.last_func_alpha) > EPS
            # calculate the trial primal and state vectors
            self.x_trial.equals_ax_p_by(1.0, self.x_start, alpha, self.search_dir)
            self.u_trial.equals_primal_solution(self.x_trial)
            # calculate and return the raw objective function
            self.func_val = objective_value(self.x_trial, self.u_trial)
            # store last used alpha
            self.last_func_alpha = alpha

        return self.func_val

    def eval_grad(self, alpha):
        # do calculations only if alpha changed significantly
        if abs(alpha - self.last_grad_alpha) > EPS:
            # calculate the trial primal and state vectors
            self.x_trial.equals_ax_p_by(1.0, self.x_start, alpha, self.search_dir)
            self.u_trial.equals_primal_solution(self.x_trial)
            # calculate objective partial
            self.primal_work.equals_objective_partial(self.x_trial, self.u_trial)
            # add contribution from objective partial
            self.p_dot_grad = self.search_dir.inner(self.primal_work)
            # calculate adjoint
            self.adjoint_work.equals_adjoint_solution(self.x_trial, self.u_trial)
            # create dR/dX jacobian wrapper
            jacobian = dRdX(self.x_trial, self.u_trial)
            # multiply the adjoint by dR/dX^T and store into primal work
            jacobian.T.product(self.adjoint_work, self.primal_work)
            # add contribution from non-linear state changes
            self.p_dot_grad += self.search_dir.inner(self.primal_work)
            # store last used alpha
            self.last_grad_alpha = alpha

        return self.p_dot_grad
