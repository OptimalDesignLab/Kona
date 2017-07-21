from collections import deque
import numpy
from kona.linalg.matrices.hessian.basic import MultiSecantApprox
from kona.linalg.vectors.composite import PrimalDualVector
from kona.linalg.solvers.util import EPS

class AndersonMultiSecant(MultiSecantApprox):
    """
    Anderson-Acceleration multi-secant approximation.

    Attributes
    ----------
    init : PrimalDualVector
        init.primal is the initial design, while init.eq and init.ineq are the initial constraints
    work : PrimalDualVector
        Work vector for constructing the difference matrices
    """

    def __init__(self, vector_factory, optns=None):
        super(AndersonMultiSecant, self).__init__(vector_factory, optns)
        # the number of vectors needed is num(x_hist) + num(r_hist) + num(x_diff) + num(r_diff) +
        # num(init + work) = 2*max_stored + 2*(max_stored - 1) + 2 = 4*max_stored
        self.primal_factory.request_num_vectors(4*self.max_stored)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(4*self.max_stored)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(4*self.max_stored)
        self.init = None
        self.work = None

    def _generate_vector(self):
        """
        Create appropriate vector based on vector factory
        """
        assert self.primal_factory is not None, \
            'AndersonMultiSecant() >> primal_factory is not defined!'
        dual_eq = None
        dual_ineq = None
        primal = self.primal_factory.generate()
        if self.eq_factory is not None:
            dual_eq = self.eq_factory.generate()
        if self.ineq_factory is not None:
            dual_ineq = self.ineq_factory.generate()
        return PrimalDualVector(primal, eq_vec=dual_eq, ineq_vec=dual_ineq)

    def set_initial_data(self, init_data):
        """
        Defines initial design (self.init.primal) and constraints (self.init.eq, self.init.ineq)
        """
        self.init = self._generate_vector()
        self.init.equals(init_data)
        self.work = self._generate_vector()

    def add_to_history(self, x_new, r_new):
        assert len(self.x_hist) == len(self.r_hist), \
            'AndersonMultiSecant() >> inconsistent list sizes!'
        if len(self.x_hist) == self.max_stored:
            # we are out of memory, so pop the oldest vectors
            self.x_hist.popleft()
            self.r_hist.popleft()
        # get new vectors to store the correction
        x = self._generate_vector()
        r = self._generate_vector()
        # copy the input vectors
        x.equals(x_new)
        r.equals(r_new)
        # insert into the lists
        self.x_hist.append(x)
        self.r_hist.append(r)

    def clear_history(self):
        """
        Clears data from x_hist and r_hist deques
        """
        self.x_hist = deque()
        self.r_hist = deque()

    def build_difference_matrices(self, alpha=0.0):
        """
        Constructs the difference matrices based on first-order optimality

        Parameters
        ----------
        alpha : Float
            controls the amount of Hessian regularization
        """
        assert len(self.x_hist) == len(self.r_hist), \
            'AndersonMultiSecant() >> inconsistent list sizes!'
        assert alpha >= 0.0, \
            'AndersonMultiSecant() >> alpha must be non-negative in build_difference_matrices!'
        # clear the previous x_diff and r_diff lists
        del self.x_diff[:], self.r_diff[:]
        self.work.equals_primaldual_residual(self.r_hist[0], self.x_hist[0].ineq)
        if alpha > EPS:
            # include Hessian regularization
            self.work.primal.equals_ax_p_by(1.0, self.work.primal, alpha, self.x_hist[0].primal)
        for k in range(len(self.x_hist)-1):
            # generate new vectors for x_diff and r_diff lists
            dx = self._generate_vector()
            dr = self._generate_vector()
            self.x_diff.append(dx)
            self.r_diff.append(dr)
            # now get the differences
            self.x_diff[k].equals_ax_p_by(1.0, self.x_hist[k+1], -1.0, self.x_hist[k])
            self.r_diff[k].equals(self.work)
            self.work.equals_primaldual_residual(self.r_hist[k+1], self.x_hist[k+1].ineq)
            if alpha > EPS:
                # include Hessian regularization
                self.work.primal.equals_ax_p_by(1.0, self.work.primal, alpha,
                                                self.x_hist[k+1].primal)
            self.r_diff[k].minus(self.work)
            self.r_diff[k].times(-1.0)

    def build_difference_matrices_for_homotopy(self, mu=0.0):
        """
        Constructs the solution and homotopy residual differences from the history

        Parameters
        ----------
        mu : float
            Homotopy continuation parameter
        """
        assert len(self.x_hist) == len(self.r_hist), \
            'AndersonMultiSecant() >> inconsistent list sizes!'
        # clear the previous x_diff and r_diff lists
        del self.x_diff[:], self.r_diff[:]
        self.work.equals_homotopy_residual(self.r_hist[0], self.x_hist[0], self.init, mu=mu)
        for k in range(len(self.x_hist)-1):
            # generate new vectors for x_diff and r_diff lists
            dx = self._generate_vector()
            dr = self._generate_vector()
            self.x_diff.append(dx)
            self.r_diff.append(dr)
            # now get the differences
            self.x_diff[k].equals_ax_p_by(1.0, self.x_hist[k+1], -1.0, self.x_hist[k])
            self.r_diff[k].equals(self.work)
            self.work.equals_homotopy_residual(self.r_hist[k+1], self.x_hist[k+1], self.init, mu=mu)
            self.r_diff[k].minus(self.work)
            self.r_diff[k].times(-1.0)

    def solve(self, in_vec, out_vec, beta=1.0, precond=None, rel_tol=1e-15):
        # store the difference matrices and rhs in numpy array format
        nvar = in_vec.get_num_var()
        dR = numpy.empty((nvar, len(self.x_diff)))
        dX = numpy.empty_like(dR)
        for k, vec in enumerate(self.r_diff):
            vec.get_base_data(dR[:,k])
        for k, vec in enumerate(self.x_diff):
            vec.get_base_data(dX[:,k])
        rhs = numpy.empty((nvar))
        in_vec.get_base_data(rhs)
        dRinv = numpy.linalg.pinv(dR, rcond=1e-6)
        sol = numpy.zeros_like(rhs)
        # sol[:] = -beta*rhs - numpy.matmul(dX - beta*dR, numpy.matmul(dRinv,rhs))
        # out_vec.set_base_data(sol)
        dRinv_r = numpy.zeros_like(rhs)
        dRinv_r = numpy.matmul(dRinv,rhs)
        sol = numpy.matmul(dR, dRinv_r) - rhs
        # move the base data into work so it can be preconditioned
        self.work.set_base_data(sol)
        if precond is None:
            out_vec.equals(self.work)
        else:
            precond(self.work, out_vec)
        out_vec.times(beta)
        sol = numpy.matmul(dX, dRinv_r)
        self.work.set_base_data(sol)
        out_vec.equals_ax_p_by(1., out_vec, -1., self.work)

# imports at the bottom to prevent circular import errors
