import numpy

from kona.user import UserSolver


class Constrained2x2(UserSolver):

    def __init__(self):
        super(Constrained2x2, self).__init__(2,2,1)

        self.dRdX = -1*numpy.eye(2)
        self.dRdU = numpy.array([[1,1],[0,1]])

    def eval_obj(self, at_design, at_state):
        return numpy.sum(at_state.data**2) + numpy.sum(at_design.data**2) - 3

    def eval_residual(self, at_design, at_state, store_here):
        p = at_design.data
        u = at_state.data
        store_here.data[:] = self.dRdU.dot(u) - p

    def eval_ceq(self, at_design, at_state, store_here):
        x = at_design.data[0]
        y = at_design.data[1]

        store_here.data[0] = x**2 + y**2

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdX.dot(in_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdU.dot(in_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdX.T.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdU.T.dot(in_vec.data)

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        x = at_design.data[0]
        y = at_design.data[1]
        out_vec.data[0] = 2*(x*in_vec.data[0] + y*in_vec.data[1])
        pass

    def multiply_dCdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = 0.0

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        x = at_design.data[0]
        y = at_design.data[1]
        out_vec.data[0] = 2*x*in_vec.data[0]
        out_vec.data[1] = 2*y*in_vec.data[0]

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = 0.0

    def eval_dFdX(self, at_design, at_state, store_here):
        X = at_design.data
        store_here.data = numpy.array([2*X[0], 2*X[1]])

    def eval_dFdU(self, at_design, at_state, store_here):
        U = at_state.data
        store_here.data = numpy.array([2*U[0], 2*U[1]])

    def init_design(self, store_here):
        store_here.data = numpy.array([10., 10.])

    def solve_nonlinear(self, at_design, result):
        result.data = numpy.linalg.solve(self.dRdU, at_design.data)
        return 0

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data = numpy.linalg.solve(self.dRdU, rhs_vec.data)
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data = numpy.linalg.solve(self.dRdU.T, rhs_vec.data)
        return 0
