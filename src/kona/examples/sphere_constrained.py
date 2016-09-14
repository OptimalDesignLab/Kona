import numpy as np

from kona.user import UserSolver

class SphereConstrained(UserSolver):

    def __init__(self, init_x=[0.51, 0.52, 0.53], ineq=False):
        self.ineq = ineq
        if not self.ineq:
            super(SphereConstrained, self).__init__(
                num_design=3,
                num_state=0,
                num_eq=1,
                num_ineq=0)
        else:
            super(SphereConstrained, self).__init__(
                num_design=3,
                num_state=0,
                num_eq=0,
                num_ineq=1)
        self.init_x = np.array(init_x)

    def eval_obj(self, at_design, at_state):
        return at_design[0] + at_design[1] + at_design[2]

    def eval_residual(self, at_design, at_state, store_here):
        store_here.data[:] = 0.

    def eval_cnstr(self, at_design, at_state):
        x = at_design[0]
        y = at_design[1]
        z = at_design[2]
        return np.array([3 - (x ** 2 + y ** 2 + z ** 2)])

    def eval_eq_cnstr(self, at_design, at_state):
        if not self.ineq:
            return self.eval_cnstr(at_design, at_state)
        else:
            return np.zeros(self.num_eq)

    def eval_ineq_cnstr(self, at_design, at_state):
        if self.ineq:
            return self.eval_cnstr(at_design, at_state)
        else:
            return np.zeros(self.num_ineq)

    def multiply_dCdX(self, at_design, at_state, in_vec):
        x = at_design[0]
        y = at_design[1]
        z = at_design[2]
        return np.array([-2 * (x * in_vec[0] + y * in_vec[1] + z * in_vec[2])])

    def multiply_dCdX_T(self, at_design, at_state, in_vec):
        x = at_design[0]
        y = at_design[1]
        z = at_design[2]
        out = np.zeros(self.num_design)
        out[0] = -2 * x * in_vec[0]
        out[1] = -2 * y * in_vec[0]
        out[2] = -2 * z * in_vec[0]
        return out

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        if not self.ineq:
            return self.multiply_dCdX(at_design, at_state, in_vec)
        else:
            return np.zeros(self.num_eq)

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        if not self.ineq:
            return self.multiply_dCdX_T(at_design, at_state, in_vec)
        else:
            return np.zeros(self.num_design)

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        if self.ineq:
            return self.multiply_dCdX(at_design, at_state, in_vec)
        else:
            return np.zeros(self.num_eq)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        if self.ineq:
            return self.multiply_dCdX_T(at_design, at_state, in_vec)
        else:
            return np.zeros(self.num_design)

    def eval_dFdX(self, at_design, at_state):
        return np.ones(self.num_design)

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data[:] = 0.0

    def init_design(self):
        return self.init_x
