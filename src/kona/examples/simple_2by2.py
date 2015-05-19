import numpy

from kona.user import UserSolver
from kona.user import BaseVector


class Simple2x2(UserSolver):

    def __init__(self):
        super(Simple2x2, self).__init__(2,2,0)

        self.dRdX = -1*numpy.eye(2)
        self.dRdU = numpy.array([[1,1],[0,1]])

    def eval_obj(self, at_design, at_state, ):
        return numpy.sum(at_state.data**2) + (at_design.data[0]-3)**2

    def eval_residual(self, at_design, at_state, store_here):
        p = at_design.data
        u = at_state.data
        store_here.data[:] = self.dRdU.dot(u) - p

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdX.dot(in_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdU.dot(in_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdX.T.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdU.T.dot(in_vec.data)

    def eval_dFdX(self, at_design, at_state, store_here):
        X = at_design.data
        store_here.data = numpy.array([2*X[0], 2*X[1]])

    def eval_dFdU(self, at_design, at_state, store_here):
        U = at_state.data
        store_here.data = numpy.array([2*(U[0]), 0])

    def init_design(self, store_here):
        store_here.data = numpy.array([10., 10.])

    def solve_nonlinear(self, at_design, result):
        result.data = numpy.linalg.solve(self.dRdU, at_design.data)
        return 0

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        X = at_design.data
        result.data = numpy.linalg.solve(dRdU, rhs_vec.data)
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data = numpy.linalg.solve(self.dRdU.T, rhs_vec.data)
        return 0

    # def user_info(self, curr_design, curr_state, curr_adj, curr_dual, num_iter):
    #     pass
