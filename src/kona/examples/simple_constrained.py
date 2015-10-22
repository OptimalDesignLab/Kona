from kona.user import UserSolver

class SimpleConstrained(UserSolver):

    def __init__(self, init_x=[0.51, 0.52, 0.53], ineq=False):
        super(SimpleConstrained, self).__init__(3, 0, 1)
        self.ineq = ineq
        self.init_x = init_x

    def eval_obj(self, at_design, at_state):
        return at_design.data[0] + at_design.data[1] + at_design.data[2]

    def eval_residual(self, at_design, at_state, store_here):
        pass

    def eval_constraints(self, at_design, at_state, store_here):
        x = at_design.data[0]
        y = at_design.data[1]
        z = at_design.data[2]
        store_here.data[0] = x**2 + y**2 + z**2 - 3

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        x = at_design.data[0]
        y = at_design.data[1]
        z = at_design.data[2]
        out_vec.data[0] = 2*(x*in_vec.data[0] + y*in_vec.data[1] +
                             z*in_vec.data[2])

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        x = at_design.data[0]
        y = at_design.data[1]
        z = at_design.data[2]
        out_vec.data[0] = 2*x*in_vec.data[0]
        out_vec.data[1] = 2*y*in_vec.data[0]
        out_vec.data[2] = 2*z*in_vec.data[0]

    def eval_dFdX(self, at_design, at_state, store_here):
        store_here.data[:] = 1.0

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data[:] = 0.0

    def init_design(self, store_here):
        store_here.data[0] = self.init_x[0]
        store_here.data[1] = self.init_x[1]
        store_here.data[2] = self.init_x[2]

    def current_solution(self, curr_design, curr_state, curr_adj,
                         curr_dual, num_iter):
        super(SimpleConstrained, self).current_solution(
            curr_design, curr_state, curr_adj, curr_dual, num_iter)

        # print self.curr_design

    def restrict_dual(self, dual_vector):
        if self.ineq:
            dual_vector.data[:] = 0.
        else:
            pass
