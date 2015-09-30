import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from kona.user import UserSolver

import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

class InversePoisson(UserSolver):

    def setup(self, nx, ny, x_control, T_control, init_array, rel_tol):
        num_design = nx - 2
        num_state = (nx-2)*(ny-2)

        super(InversePoisson, self).__init__(num_design, num_state, 0)

        self.Nx = nx-2
        self.Ny = ny-2
        self.hx = 1./(self.Nx+1)
        self.hy = 1./(self.Ny+1)

        self.start_from = init_array
        self.des_info = True
        self.solve_info = True
        self.solve_calls = 0
        self.precond_calls = 0
        self.Kvcount = 0
        self.precond_calls_total = 0

        self.x_control = x_control
        self.T_control = T_control

        self.rel_tol = rel_tol
        T, converged = self.solve_state(self.T_control, self.rel_tol)

        if converged:
            self.T_target = T
        else:
            print "solve state not converged! setup"

    def add_solves(self):
        self.solve_calls += 1

    def reset_solves(self):
        self.solve_calls = 0

    def eval_obj(self, at_design, at_state):
        if at_state == -1:
            x = at_design.data
            T, converged = self.solve_state(x, self.rel_tol)
            if not converged:
                precond = -self.precond_calls
            else:
                precond = self.precond_calls
        else:
            T = at_state.data
            precond = 0

        diff_T = (T - self.T_target)
        self.obj = (diff_T.dot(diff_T))*(self.hx*self.hy)
        print "current obj:", self.obj
        return (self.obj, precond)

    def eval_residual(self, at_design, at_state, store_here):
        x = at_design.data
        T = at_state.data
        b_res = np.zeros(self.Ny * self.Nx)
        b_res[:self.Nx] =  (1./(self.hy)**2) * x
        store_here.data = self.apply_Kv(T) - b_res

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        z = in_vec.data
        temp = np.zeros(self.Ny*self.Nx)
        temp[:self.Nx] = (-1.)/((self.hy)**2) * z
        out_vec.data = temp

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        z = in_vec.data
        out_vec.data = self.apply_Kv(z)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        z = in_vec.data
        out_vec.data = (-1.)/((self.hy)**2) * z[:self.Nx]

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        z = in_vec.data
        out_vec.data = self.apply_Kv(z)

    def eval_dFdX(self, at_design, at_state, store_here):
        store_here.data = np.zeros(self.Nx)

    def eval_dFdU(self, at_design, at_state, store_here):
        T = at_state.data
        diff_T = T - self.T_target
        store_here.data = 2 * (self.hx * self.hy) * diff_T

    def init_design(self, store_here):
        store_here.data = self.start_from

    def solve_nonlinear(self, at_design, store_here):
        x = at_design.data
        T, converged = self.solve_state(x, self.rel_tol)
        store_here.data = T
        if converged:
            if self.solve_info:
                print "System solve CONVERGED!"
            return self.precond_calls
        else:
            print "System solve FAILED!"
            return -self.precond_calls

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        b_rhs = rhs_vec.data
        T, converged = self.solve_state(b_rhs, rel_tol)
        result.data = T
        if converged:
            return self.precond_calls
        else:
            print "solve_state not converged, solve_linear!"
            return -self.precond_calls

    def solve_adjoint(self, at_design, at_state, rhs_vec, tol, result):
        b_rhs = rhs_vec.data
        T, converged = self.solve_state(b_rhs, tol)
        result.data = T
        if converged:
            return self.precond_calls
        else:
            return -self.precond_calls

    def current_solution(self, curr_design, curr_state, curr_adj,
                         curr_dual, num_iter):
        # print the current design and state vectors at each iteration
        self.curr_design = curr_design.data
        self.num_iter = num_iter

    def plot_field(self, nx, ny, T_control, T_field):

        Ny = ny - 2
        Nx = nx - 2

        T = np.zeros((ny,nx))
        T[0, 1:-1] = T_control
        T[1:-1, 1:-1] = T_field.reshape((Ny, Nx))

        X = np.linspace(0.0, 1.0, num=nx)
        Y = np.linspace(0.0, 1.0, num=ny)
        X, Y = np.meshgrid(X, Y)

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        surf1 = ax1.plot_surface(
            X, Y, T, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.title('Target T')
        ax1.set_zlim(0, 0.3)
        plt.colorbar(surf1, shrink=0.5, aspect=5)
        ax1.view_init(30,45)
        plt.savefig('temp.png')

    def solve_state(self, design, rel_tol):
        """
        parameters for CG the KrylovSolver:

        Parameters
        ----------
        maxiter   : int
            maximum number of allowed iterations
        tol       : float
            relative convergence tolerance, i.e. tol is scaled by ||b||
        residuals : list
            residuals has the residual norm history, including the initial
            residual, appended to it
        """

        if len(design) == (self.Nx*self.Ny):
            # when the input vector is of the same length as the state vector
            # it can be used as the RHS of the system eqaution Kv = b
            # granted that the input vector is a valid and correct RHS, b.c.
            # included
            b = design
        else:
            # when the input vector is of design vector's length
            # assemble it to the RHS vector

            b = np.zeros(self.Ny*self.Nx)
            b[:self.Nx] = (1./(self.hy)**2) * design

        self.reset_precond()

        Kv = LinearOperator(
            (self.Nx*self.Ny, self.Nx*self.Ny),
            matvec=lambda v: self.apply_Kv(v), dtype='float64')

        max_iter = 2*self.Nx*self.Ny
        converged = False

        (T, info) = cg(Kv, b, maxiter=max_iter, tol=rel_tol, residuals=None)
        # 0 : successful exit
        # >0: convergence to tolerance not achieved, return iteration count

        self.add_solves()

        if info == 0:
            converged = True
            print "Preconditioned CG converged"
        else:
            raise TypeError("solve_state() >> CG failed!")
        return T, converged

    def apply_Kv(self, v):
        """
        Matrix free method for calculating the 2D poisson stiffness matrix's
        product with a vector

        """
        self.Kvcount += 1

        v_mat = v.reshape((self.Ny, self.Nx))
        Dh1v = np.zeros((self.Ny, self.Nx))

        for k in np.arange(self.Ny):
            Dh1v[k,] = self.apply_Dh1_v(v_mat[k,])

        Dh1v = Dh1v.flatten()

        Dh2v = np.zeros((self.Ny, self.Nx))
        Dh2v[0,] = 2*v_mat[0,] - v_mat[1,]
        Dh2v[-1,] = -v_mat[-2,] + 2*v_mat[-1,]
        Dh2v[1:-1,] = -v_mat[:-2, ] + 2*v_mat[1:-1,] - v_mat[2:,]
        Dh2v = Dh2v.flatten()

        Kv = 1./(self.hx**2)*Dh1v + 1./(self.hy**2)*Dh2v
        return Kv

    def apply_Dh1_v(self, v):

        Dh1v_sub = np.zeros(len(v))

        Dh1v_sub[0] = 2*v[0] - v[1]
        Dh1v_sub[-1] = -v[-2] + 2*v[-1]
        Dh1v_sub[1:-1] = -v[:-2] + 2*v[1:-1] - v[2:]

        return Dh1v_sub

    def apply_Kdotv(self, v):
        pass

    def apply_KTv(self, v):

        self.apply_Kv(v)    # because K is symmetrical in this case
