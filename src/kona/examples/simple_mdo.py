import numpy as np

from kona.user import UserSolver, UserSolverIDF

class CoreMDO(object):

    def __init__(self, num_disc=5, init_x=5):
        self.num_disc = num_disc

        self.A = np.zeros((self.num_disc, self.num_disc))
        self.A_inv = np.zeros((self.num_disc, self.num_disc))

        for i in range(self.num_disc):
            self.A[i, i] = -2.
            self.A_inv[i, i] = -0.5
            if i == 0:
                self.A[i, i + 1] = 1.
            elif i == self.num_disc - 1:
                self.A[i, i - 1] = 1.
            else:
                self.A[i, i + 1] = 1.
                self.A[i, i - 1] = 1.

        self.e1 = np.zeros((self.num_disc, 1))
        self.e1[0] = 1.
        self.em = np.zeros((self.num_disc, 1))
        self.em[-1] = 1.

        self.eme1T = self.em.dot(self.e1.T)
        self.e1emT = self.e1.dot(self.em.T)

class SimpleMDF(UserSolver):

    def __init__(self, num_disc=5, init_x=5.):
        self.mdo = CoreMDO(num_disc)
        self.num_disc = self.mdo.num_disc
        self.A = self.mdo.A
        self.e1 = self.mdo.e1
        self.em = self.mdo.em
        self.eme1T = self.mdo.eme1T
        self.e1emT = self.mdo.e1emT

        super(SimpleMDF, self).__init__(
            1, self.num_disc*2,
            num_eq=0,
            num_ineq=0)

        top_half = np.concatenate((self.A, self.eme1T), axis=1)
        bottom_half = np.concatenate((self.e1emT, self.A), axis=1)
        self.system = np.concatenate((top_half, bottom_half), axis=0)

        self.init_x = np.array([init_x])

    def eval_obj(self, at_design, at_state):
        u1 = at_state.data[:self.num_disc]
        u2 = at_state.data[self.num_disc:]
        J = 0.
        for i in range(self.num_disc-1):
            J += (u1[i+1] - u1[i])**2 + (u2[i+1] - u2[i])**2
        return J

    def eval_dFdX(self, at_design, at_state):
        return np.zeros(1)

    def eval_dFdU(self, at_design, at_state, store_here):
        u1 = at_state.data[:self.num_disc]
        u2 = at_state.data[self.num_disc:]
        dfdu = np.zeros(self.num_state)
        for i in range(self.num_disc-1):
            dfdu[i] += 2*(u1[i+1] - u1[i])*(-1)
            dfdu[i+1] += 2*(u1[i+1] - u1[i])
            dfdu[self.num_disc+i] += 2*(u2[i+1] - u2[i])*(-1)
            dfdu[self.num_disc+i+1] += 2*(u2[i+1] - u2[i])
        store_here.data[:]= dfdu[:]

    def eval_residual(self, at_design, at_state, store_here):
        x = at_design[0]
        u1 = at_state.data[:self.num_disc]
        u2 = at_state.data[self.num_disc:]
        xe1 = x*self.e1.reshape((self.num_disc,))
        store_here.data[:self.num_disc] = \
            self.A.dot(u1) + self.eme1T.dot(u2) - xe1
        store_here.data[self.num_disc:] = \
            self.e1emT.dot(u1) + self.A.dot(u2)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        x = in_vec[0]
        xe1 = x*self.e1.reshape((self.num_disc,))
        out_vec.data[:self.num_disc] = -xe1[:]
        out_vec.data[self.num_disc:] = 0.

    def multiply_dRdX_T(self, at_design, at_state, in_vec):
        u1_0 = in_vec.data[0]
        return np.array([-u1_0])

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.system.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.system.T.dot(in_vec.data)

    def solve_nonlinear(self, at_design, result):
        rhs = np.zeros(self.num_state)
        rhs[0] = at_design[0]
        result.data[:] = np.linalg.solve(self.system, rhs)
        return 1

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = np.linalg.solve(self.system, rhs_vec.data)
        return 1

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = np.linalg.solve(self.system.T, rhs_vec.data)
        return 1

    def init_design(self):
        return self.init_x

class SimpleIDF(UserSolverIDF):

    def __init__(self, num_disc=5, init_x=5., approx_inv=True):
        self.mdo = CoreMDO(num_disc)
        self.num_disc = self.mdo.num_disc
        self.A = self.mdo.A
        self.A_inv = self.mdo.A_inv
        self.e1 = self.mdo.e1
        self.em = self.mdo.em
        self.eme1T = self.mdo.eme1T
        self.e1emT = self.mdo.e1emT

        super(SimpleIDF, self).__init__(
            1, self.num_disc*2, 2,
            num_eq=0,
            num_ineq=0)

        empty = np.zeros((self.num_disc, self.num_disc))
        top_half = np.concatenate((self.A, empty), axis=1)
        bottom_half = np.concatenate((empty, self.A), axis=1)
        self.system = np.concatenate((top_half, bottom_half), axis=0)
        if approx_inv:
            top_inv = np.concatenate((self.A_inv, empty), axis=1)
            bottom_inv = np.concatenate((empty, self.A_inv), axis=1)
            self.system_inv = np.concatenate((top_inv, bottom_inv), axis=0)
        else:
            self.system_inv = np.linalg.inv(self.system)

        self.init_x = np.array([init_x, 1., 1.])

    def eval_obj(self, at_design, at_state):
        u1 = at_state.data[:self.num_disc]
        u2 = at_state.data[self.num_disc:]
        J = 0.
        for i in range(self.num_disc-1):
            J += (u1[i+1] - u1[i])**2 + (u2[i+1] - u2[i])**2
        return J

    def eval_dFdX(self, at_design, at_state):
        return np.zeros(3)

    def eval_dFdU(self, at_design, at_state, store_here):
        u1 = at_state.data[:self.num_disc]
        u2 = at_state.data[self.num_disc:]
        dfdu = np.zeros(self.num_state)
        for i in range(self.num_disc-1):
            dfdu[i] += 2*(u1[i+1] - u1[i])*(-1)
            dfdu[i+1] += 2*(u1[i+1] - u1[i])
            dfdu[self.num_disc+i] += 2*(u2[i+1] - u2[i])*(-1)
            dfdu[self.num_disc+i+1] += 2*(u2[i+1] - u2[i])
        store_here.data[:]= dfdu[:]

    def eval_residual(self, at_design, at_state, store_here):
        x = at_design[0]
        u1t = at_design[1]
        u2t = at_design[2]
        u1 = at_state.data[:self.num_disc]
        u2 = at_state.data[self.num_disc:]
        xe1 = x*self.e1.reshape((self.num_disc,))
        store_here.data[:self.num_disc] = \
            self.A.dot(u1) + xe1 + u2t*self.em.reshape((self.num_disc,))
        store_here.data[self.num_disc:] = \
            u1t*self.e1.reshape((self.num_disc,)) + self.A.dot(u2)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        x = in_vec[0]
        u1t = in_vec[1]
        u2t = in_vec[2]
        xe1 = x*self.e1.reshape((self.num_disc,))
        out_vec.data[:self.num_disc] = \
            xe1 + u2t*self.em.reshape((self.num_disc,))
        out_vec.data[self.num_disc:] = \
            u1t * self.e1.reshape((self.num_disc,))

    def multiply_dRdX_T(self, at_design, at_state, in_vec):
        u1 = in_vec.data[:self.num_disc]
        u2 = in_vec.data[self.num_disc:]
        out = np.zeros(3)
        out[0] = u1[0]
        out[1] = u2[0]
        out[2] = u1[-1]
        return out

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.system.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        self.multiply_dRdU(at_design, at_state, in_vec, out_vec)

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = self.system_inv.dot(in_vec.data)

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        self.apply_precond(at_design, at_state, in_vec, out_vec)

    def eval_eq_cnstr(self, at_design, at_state):
        u1t = at_design[1]
        u2t = at_design[2]
        u1 = at_state.data[:self.num_disc]
        u2 = at_state.data[self.num_disc:]
        out = np.zeros(2)
        out[0] = u1[-1] - u1t
        out[1] = u2[0] - u2t
        return out

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        out = np.zeros(2)
        out[0] = -in_vec[1]
        out[1] = -in_vec[2]
        return out

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        out = np.zeros(3)
        out[0] = 0.
        out[1] = -in_vec[0]
        out[2] = -in_vec[1]
        return out

    def multiply_dCEQdU(self, at_design, at_state, in_vec):
        u1 = in_vec.data[:self.num_disc]
        u2 = in_vec.data[self.num_disc:]
        out = np.zeros(2)
        out[0] = u1[-1]
        out[1] = u2[0]
        return out

    def multiply_dCEQdU_T(self, at_design, at_state, in_vec, out_vec):
        c1 = in_vec[0]
        c2 = in_vec[1]
        out_vec.data[:] = 0.
        out_vec.data[:self.num_disc][-1] = c1
        out_vec.data[self.num_disc:][0] = c2

    def solve_nonlinear(self, at_design, result):
        x = at_design[0]
        u1t = at_design[1]
        u2t = at_design[2]
        rhs = np.zeros(self.num_state)
        xe1 = x * self.e1.reshape((self.num_disc,))
        rhs[:self.num_disc] = -xe1 - u2t*self.em.reshape((self.num_disc,))
        rhs[self.num_disc:] = -u1t*self.e1.reshape((self.num_disc,))
        result.data[:] = np.linalg.solve(self.system, rhs)
        return 1

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = np.linalg.solve(self.system, rhs_vec.data)
        return 1

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        return self.solve_linear(at_design, at_state, rhs_vec, rel_tol, result)

    def init_design(self):
        return self.init_x
