import numpy as np

from kona.user import UserSolver

class SpiralSolver(object):

    def linearize(self, at_design, at_state=None):
        self.X = at_design.data[0]
        if at_state is not None:
            self.U = at_state.data
        else:
            self.U = None

    @property
    def theta(self):
        return 0.5*(self.X + np.pi)

    @property
    def alpha(self):
        return 0.5*(self.X - np.pi)

    @property
    def dRdX(self):
        dRdX1 = -np.sin(self.theta)*self.U[0] + np.cos(self.theta)*self.U[1] \
            + (self.X**2)*np.sin(self.alpha) - 4.0*self.X*np.cos(self.alpha)
        dRdX2 = -np.cos(self.theta)*self.U[0] - np.sin(self.theta)*self.U[1] \
            - (self.X**2)*np.cos(self.alpha) - 4.0*self.X*np.sin(self.alpha)
        return 0.5*np.array([[dRdX1],[dRdX2]])

    @property
    def dRdU(self):
        return np.array([[np.cos(self.theta), np.sin(self.theta)],
                         [-np.sin(self.theta), np.cos(self.theta)]])

    @property
    def rhs(self):
        return np.array([(self.X**2)*np.cos(self.alpha),
                         (self.X**2)*np.sin(self.alpha)])

    @property
    def R(self):
        return self.dRdU.dot(self.U) - self.rhs

    @property
    def F(self):
        return 0.5*(self.X**2 + self.U[0]**2 + self.U[1]**2)

    @property
    def dFdX(self):
        return np.array([self.X])

    @property
    def dFdU(self):
        return np.array([self.U[0], self.U[1]])


class Spiral(UserSolver):

    def __init__(self):
        super(Spiral, self).__init__(1,2,0)
        self.PDE = SpiralSolver()
        self.x_hist = []
        self.u1_hist = []
        self.u2_hist = []
        self.obj_hist = []
        self.grad_hist = []

    def eval_obj(self, at_design, at_state):
        self.PDE.linearize(at_design, at_state)
        return self.PDE.F

    def eval_residual(self, at_design, at_state, store_here):
        self.PDE.linearize(at_design, at_state)
        store_here.data = self.PDE.R

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        self.PDE.linearize(at_design, at_state)
        out_vec.data = self.PDE.dRdX.dot(in_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        self.PDE.linearize(at_design, at_state)
        out_vec.data = self.PDE.dRdU.dot(in_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        self.PDE.linearize(at_design, at_state)
        out_vec.data = self.PDE.dRdX.T.dot(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        self.PDE.linearize(at_design, at_state)
        out_vec.data = self.PDE.dRdU.T.dot(in_vec.data)

    def eval_dFdX(self, at_design, at_state, store_here):
        self.PDE.linearize(at_design, at_state)
        store_here.data = self.PDE.dFdX

    def eval_dFdU(self, at_design, at_state, store_here):
        self.PDE.linearize(at_design, at_state)
        store_here.data = self.PDE.dFdU

    def init_design(self, store_here):
        store_here.data = np.array([8.0*np.pi])

    def solve_nonlinear(self, at_design, result):
        # start with initial guess for solution and linearize
        self.PDE.linearize(at_design)
        self.PDE.U = np.zeros(2)
        # calculate initial residual
        rel_tol = 1e-8
        norm0 = np.linalg.norm(self.PDE.R)
        max_iter = 100
        converged = False
        # start Newton loop
        for iters in xrange(max_iter):
            # check convergence with new residual
            if np.linalg.norm(self.PDE.R) <= rel_tol*norm0:
                converged = True
                break
            # run linear solution and update state variables
            self.PDE.U += np.linalg.solve(self.PDE.dRdU, -self.PDE.R)
        # write result and return cost
        result.data = self.PDE.U
        if converged:
            return iters
        else:
            return -iters

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        self.PDE.linearize(at_design, at_state)
        result.data = np.linalg.solve(self.PDE.dRdU, rhs_vec.data)
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        self.PDE.linearize(at_design, at_state)
        result.data = np.linalg.solve(self.PDE.dRdU.T, rhs_vec.data)
        return 0

    def current_solution(self, curr_design, curr_state, curr_adj,
                         curr_dual, num_iter):

        super(Spiral, self).current_solution(
            curr_design, curr_state, curr_adj, curr_dual, num_iter)

        print 'Current Design: '
        print self.curr_design
        print 'Current State: '
        print self.curr_state

        self.PDE.linearize(curr_design, curr_state)
        self.x_hist.append(curr_design.data[0])
        self.u1_hist.append(curr_state.data[0])
        self.u2_hist.append(curr_state.data[1])
        self.obj_hist.append(self.PDE.F)
        self.grad_hist.append(
            self.PDE.dFdX + self.PDE.dRdX.T.dot(curr_adj.data))
