import numpy as np

from kona.user import UserSolver

class Constrained2x2(UserSolver):

    def __init__(self):
        super(Constrained2x2, self).__init__(2, 2, num_eq=1)

        self.dRdX = -1*np.eye(2)
        self.dRdU = np.array([[1,1],[0,1]])

    def eval_obj(self, at_design, at_state):
        return np.sum(at_state.data**2) + np.sum(at_design**2) - 3

    def eval_residual(self, at_design, at_state, store_here):
        p = at_design
        u = at_state.data
        store_here.data[:] = self.dRdU.dot(u) - p

    def eval_eq_cnstr(self, at_design, at_state, store_here):
        x = at_design[0]
        y = at_design[1]

        return np.array([x**2 + y**2])

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdX.dot(in_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdU.dot(in_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        return self.dRdX.T.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data = self.dRdU.T.dot(in_vec.data)

    def multiply_dCEQdX(self, at_design, at_state, in_vec, out_vec):
        x = at_design[0]
        y = at_design[1]
        return np.array([2*(x*in_vec[0] + y*in_vec[1])])

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec, out_vec):
        x = at_design[0]
        y = at_design[1]
        return np.array([2*x*in_vec[0], 2*y*in_vec[0]])

    def multiply_dCEQdU(self, at_design, at_state, in_vec, out_vec):
        return np.zeros(self.num_eq)

    def multiply_dCEQdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = 0.0

    def eval_dFdX(self, at_design, at_state, store_here):
        X = at_design
        return np.array([2*X[0], 2*X[1]])

    def eval_dFdU(self, at_design, at_state, store_here):
        U = at_state.data
        store_here.data = np.array([2*U[0], 2*U[1]])

    def init_design(self, store_here):
        return np.array([10., 10.])

    def solve_nonlinear(self, at_design, result):
        result.data = np.linalg.solve(self.dRdU, at_design)
        return 1

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data = np.linalg.solve(self.dRdU, rhs_vec.data)
        return 1

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data = np.linalg.solve(self.dRdU.T, rhs_vec.data)
        return 1
