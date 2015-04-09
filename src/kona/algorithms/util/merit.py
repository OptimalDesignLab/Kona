import sys

from kona.linalg.vectors.common import objective_value

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
        self.primal_factory.request_num_vectors(1)

        self.state_factory = state_factory
        self.state_factory.request_num_vectors(1)

        self.out_file = out_file

        self.x_trial = None
        self.y_trial = None

    def reset(self, p, x, p_dot_grad=None, state=None, adjoint=None):
        if self.x_trial is None:
            self.x_trial = self.primal_factory.generate()
            self.u_trial = self.state_factory.generate()

        self.x_start = x
        self.search_dir = p

    def eval_func(self, alpha):
        self.x_trial.equals_ax_p_by(1.0, self.x_start, alpha,
            self.search_dir)
        self.u_trial.equals_primal_solution(self.x_trial)

        return objective_value(self.x_trial, self.u_trial)
