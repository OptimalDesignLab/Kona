import numpy

from kona.user import UserSolver

class Simple2x2(UserSolver):

    def __init__(self):
        super(Simple2x2, self).__init__(
            num_design=2,
            num_state=2,
            num_eq=0,
            num_ineq=0)

        self.dRdX = -1*numpy.eye(2)
        self.dRdU = numpy.array([[1,1],[0,1]])

    def eval_obj(self, at_design, at_state):
        return numpy.sum(at_state.data**2) + (at_design[0]-3)**2

    def eval_residual(self, at_design, at_state, store_here):
        p = at_design
        u = at_state.data
        store_here.data[:] = self.dRdU.dot(u) - p

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.dRdX.dot(in_vec)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.dRdU.dot(in_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec):
        return self.dRdX.T.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.dRdU.T.dot(in_vec.data)

    def eval_dFdX(self, at_design, at_state):
        return numpy.array([2*at_design[0], 2*at_design[1]])

    def eval_dFdU(self, at_design, at_state, store_here):
        U = at_state.data
        store_here.data[:] = numpy.array([2*(U[0]), 0])

    def init_design(self):
        return numpy.array([10., 10.])

    def solve_nonlinear(self, at_design, result):
        result.data[:] = numpy.linalg.solve(self.dRdU, at_design)
        return 0

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = numpy.linalg.solve(self.dRdU, rhs_vec.data)
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = numpy.linalg.solve(self.dRdU.T, rhs_vec.data)
        return 0
