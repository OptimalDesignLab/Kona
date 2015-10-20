import numpy as np

from kona.user import UserSolver

class ExponentialConstrained(UserSolver):

    def __init__(self, init_x=[1., 1.]):
        super(ExponentialConstrained, self).__init__(2, 0, 1)
        self.init_x = init_x

    def eval_obj(self, at_design, at_state):
        x = at_design.data[0]
        y = at_design.data[1]
        return x + y**2

    def eval_residual(self, at_design, at_state, store_here):
        pass

    def eval_constraints(self, at_design, at_state, store_here):
        x = at_design.data[0]
        store_here.data[0] = np.exp(x) - 1.

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        x = at_design.data[0]
        out_vec.data[0] = np.exp(x)*in_vec.data[0]

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        x = at_design.data[0]
        out_vec.data[0] = np.exp(x)*in_vec.data[0]
        out_vec.data[1] = 0.

    def eval_dFdX(self, at_design, at_state, store_here):
        y = at_design.data[1]
        store_here.data[0] = 1.
        store_here.data[1] = 2*y

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data[:] = 0.0

    def init_design(self, store_here):
        store_here.data[0] = self.init_x[0]
        store_here.data[1] = self.init_x[1]

    def current_solution(self, curr_design, curr_state, curr_adj,
                         curr_dual, num_iter):
        super(ExponentialConstrained, self).current_solution(
            curr_design, curr_state, curr_adj, curr_dual, num_iter)

        # print self.curr_design

    def apply_active_set(self, at_cnstr, in_vec, out_vec):
        if at_cnstr.data[0] >= 0.:
            out_vec.data[0] = 0.
        else:
            out_vec.data[0] = in_vec.data[0]
