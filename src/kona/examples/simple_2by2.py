import numpy

from kona.user import UserSolver
from kona.user import BaseVector


class Simple2x2(UserSolver):

    def __init__(self):
        super(Simple2x2, self).__init__(4,2,0)

    def eval_obj(self, at_design, at_state, ):
        return 2*numpy.sum(at_design.data**2) + numpy.sum(at_state.data)

    def eval_residual(self, at_design, at_state, store_here):
        p = at_design.data
        A = p.reshape(2,2)
        store_here.data[:] = A.dot(at_state.data) - np.array([1,2])

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        u = at_state.data
        dRdX = np.zeros((4,2))
        dRdX[0][:2] = u
        dRdX[1][2:] = u
        out_vec.data = dRdX.dot(in_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        X = at_design.data
        dRdU = X.reshape(2,2)
        out_vec.data[:] = dRdU.dot(in_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        u = at_state.data
        dRdX = np.zeros((4,2))
        dRdX[0][:2] = u
        dRdX[1][2:] = u
        out_vec.data = dRdX.T.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        X = at_design.data
        dRdU = X.reshape(2,2)
        out_vec.data[:] = dRdU.T.dot(in_vec.data)

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
