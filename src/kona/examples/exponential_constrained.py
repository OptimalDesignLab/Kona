import numpy as np

from kona.user import UserSolver

class ExponentialConstrained(UserSolver):

    def __init__(self, init_x=[1., 1.]):
        super(ExponentialConstrained, self).__init__(
            num_design=2,
            num_state=0,
            num_eq=1,
            num_ineq=0)
        self.init_x = np.array(init_x)

    def eval_obj(self, at_design, at_state):
        x = at_design[0]
        y = at_design[1]
        return x + y**2

    def eval_residual(self, at_design, at_state, store_here):
        pass

    def eval_eq_cnstr(self, at_design, at_state):
        x = at_design[0]
        return np.array([np.exp(x) - 1.])

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        x = at_design[0]
        return np.array([np.exp(x)*in_vec[0]])

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        x = at_design[0]
        return np.array([np.exp(x)*in_vec[0], 0.])

    def eval_dFdX(self, at_design, at_state):
        y = at_design[1]
        return np.array([1., 2*y])

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data[:] = 0.0

    def init_design(self):
        return self.init_x
