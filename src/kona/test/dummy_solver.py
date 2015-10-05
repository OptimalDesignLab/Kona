import numpy

from kona.user import UserSolver

class DummySolver(UserSolver):

    def eval_obj(self, at_design, at_state):
        return numpy.sum(at_design.data) + numpy.sum(at_state.data)

    def eval_residual(self, at_design, at_state, store_here):
        store_here.data[:] = self.eval_obj(at_design, at_state)

    def eval_constraints(self, at_design, at_state, store_here):
        store_here.data[:] = self.eval_obj(at_design, at_state)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def build_precond(self):
        pass

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)
        return 0

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)
        return 0

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def multiply_dCdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(in_vec.data)

    def eval_dFdX(self, at_design, at_state, store_here):
        store_here.data[:] = self.eval_obj(at_design, at_state)

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data[:] = self.eval_obj(at_design, at_state)

    def init_design(self, store_here):
        store_here.data[:] = 10.

    def solve_nonlinear(self, at_design, result):
        result.data[:] = numpy.sum(at_design.data)
        return 0

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(rhs_vec.data)
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = self.eval_obj(at_design, at_state) + \
            numpy.sum(rhs_vec.data)
        return 0

    def user_info(self, curr_design, curr_state, curr_adj, curr_dual, num_iter):
        pass
