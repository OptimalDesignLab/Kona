import numpy as np

from kona.user import UserSolver

class Sellar(UserSolver):

    def __init__(self, init_x=[5., 2., 1.], des_bounds=True):
        super(Sellar, self).__init__(
            num_design=3,
            num_state=2,
            num_eq=0,
            num_ineq=8)
        self.init_x = np.array(init_x)
        self.des_bounds = des_bounds

        if self.des_bounds:
            self.dCdX = np.array(
                [[0., 0., 0.],
                 [0., 0., 0.],
                 [1., 0., 0.],
                 [-1., 0., 0.],
                 [0., 1., 0.],
                 [0., -1., 0.],
                 [0., 0., 1.],
                 [0., 0., -1.]]
            )
        else:
            self.dCdX = np.array(
                [[0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.],
                 [0., 0., 0.]]
            )

        self.dCdU = np.array(
            [[1./3.16, 0.],
             [0., -1./24.],
             [0., 0.],
             [0., 0.],
             [0., 0.],
             [0., 0.],
             [0., 0.],
             [0., 0.]]
        )

    def eval_obj(self, at_design, at_state):
        x2 = at_design[1]
        x3 = at_design[2]
        y1 = at_state.data[0]
        y2 = at_state.data[1]
        return (x3**2 + x2 + y1 + np.exp(-y2), 0)

    def eval_dFdX(self, at_design, at_state):
        x3 = at_design[2]
        return np.array([0., 1., 2.*x3])

    def eval_dFdU(self, at_design, at_state, store_here):
        y2 = at_state[1]
        store_here.data[0] = 1.
        store_here.data[1] = -np.exp(-y2)

    def eval_residual(self, at_design, at_state, store_here):
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        y1 = at_state.data[0]
        y2 = at_state.data[1]
        store_here.data[0] = y1 - x1**2 - x3 - x2 + 0.2*y2
        store_here.data[1] = y2 - np.sqrt(y1) - x1 - x2

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        x1 = at_design[0]
        dRdX = np.array(
            [[-2.*x1, -1., -1.],
             [-1., -1., 0.]])
        out_vec.data[:] = np.dot(dRdX, in_vec)

    def multiply_dRdX_T(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        dRdX = np.array(
            [[-2.*x1, -1., -1.],
             [-1., -1., 0.]])
        return np.dot(dRdX.T, in_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        y1 = at_state[0]
        dRdU = np.array(
            [[1., 0.2],
             [-.5*y1**-.5, 1.]])
        out_vec.data[:] = np.dot(dRdU, in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        y1 = at_state.data[0]
        dRdU = np.array(
            [[1., 0.2],
             [-.5*y1**-.5, 1.]])
        out_vec.data[:] = np.dot(dRdU.T, in_vec.data)

    def eval_ineq_cnstr(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        y1 = at_state.data[0]
        y2 = at_state.data[1]
        out = np.zeros(self.num_ineq)
        out[0] = y1/3.16 - 1.
        out[1] = 1. - y2/24.
        if self.des_bounds:
            out[2] = x1 + 10.
            out[3] = 10. - x1
            out[4] = x2
            out[5] = 10. - x2
            out[6] = x3
            out[7] = 10. - x3
        else:
            out[2:] = 0.
        return out

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        return np.dot(self.dCdX, in_vec)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        return np.dot(self.dCdX.T, in_vec)

    def multiply_dCINdU(self, at_design, at_state, in_vec):
        return np.dot(self.dCdU, in_vec.data)

    def multiply_dCINdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = np.dot(self.dCdU.T, in_vec)

    def solve_nonlinear(self, at_design, result):

        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]

        iters = 0
        max_iter = 50
        u = np.zeros(2)
        u[0] = 3.16
        u[1] = 24.
        R = np.zeros(2)
        abs_tol = 1e-6
        du = 0
        while iters < max_iter:
            y1 = u[0]
            y2 = u[1]

            if y1 < 0.:
                u -= du
                break

            R[0] = y1 - x1**2 - x3 - x2 + 0.2*y2
            R[1] = y2 - np.sqrt(y1) - x1 - x2

            if np.linalg.norm(R) < abs_tol:
                result.data[0] = u[0]
                result.data[1] = u[1]
                return iters

            dRdU = np.array(
                [[1., 0.2],
                 [-.5*y1**-.5, 1.]])

            du = np.linalg.solve(dRdU, -R)

            u += du

        result.data[0] = u[0]
        result.data[1] = u[1]
        return -1

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        y1 = at_state[0]
        dRdU = np.array(
            [[1., 0.2],
             [-.5*y1**-.5, 1.]])
        result.data = np.linalg.solve(dRdU, rhs_vec.data)
        return 1

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        y1 = at_state[0]
        dRdU = np.array(
            [[1., 0.2],
             [-.5*y1**-.5, 1.]])
        result.data = np.linalg.solve(dRdU.T, rhs_vec.data)
        return 1

    def init_design(self):
        return self.init_x
