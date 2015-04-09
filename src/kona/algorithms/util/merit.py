import sys

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

    def __init__(self, primal_factory, optns, out_file=sys.stdout):
        self.primal_factory = primal_factory
        self.out_file = out_file
        self.primal_factory.request_num_vectors(0)

    def reset(self, p, x, p_dot_grad, state, adjoint):
        self.x_start = x
        self.search_dir = p

    def eval_func(self, alpha):
        self.design_work.equals_ax_plus_by(1.0, self.x_start, alpha,
            self.search_dir)
