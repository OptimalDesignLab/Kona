import numpy

from kona.user import UserSolver
from kona.user import BaseVector

class DummySolver(UserSolver):

    def eval_obj(self, at_design, at_state, ):
        return numpy.sum(at_design.data) + numpy.sum(at_state.data)

    def eval_residual(self, at_design, at_state, store_here):
        store_here.data[:] = numpy.sum(at_design.data) + at_state.data[:]

    def eval_ceq(self, at_design, at_state, store_here):
        store_here.data[:] = self.eval_obj(at_design, at_state)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = numpy.inner(at_design.data, in_vec.data) + at_state.data[:]

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        pass

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = numpy.inner(at_state.data, in_vec.data) + at_design.data[:]

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        pass

    def build_precond(self):
        pass

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        return 0

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        return 0

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state)*numpy.sum(in_vec)

    def multiply_dCdU(self, at_design, at_state, in_vec, out_vec):
        pass

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.eval_obj(at_design, at_state)*numpy.sum(in_vec)

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        pass

    def eval_dFdX(self, at_design, at_state, store_here):
        store_here.data[:] = numpy.sum(at_state.data)*at_design.data[:]

    def eval_dFdU(self, at_design, at_state, store_here):
        pass

    def init_design(self, store_here):
        store_here.data = 4.*numpy.ones(self.num_primal)

    def solve_nonlinear(self, at_design, result):
        return 0

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        return 0

    def user_info(self, curr_design, curr_state, curr_adj, curr_dual, num_iter):
        pass
