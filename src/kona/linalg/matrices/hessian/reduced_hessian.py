import numpy

from kona.options import get_opt
from kona.linalg.vectors.common import PrimalVector, StateVector
from kona.linalg.matrices.common import dRdX, dRdU, IdentityMatrix
from kona.linalg.matrices.hessian.basic import BaseHessian, QuasiNewtonApprox
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import calc_epsilon

class ReducedHessian(BaseHessian):
    """
    Reduced-space approximation of the Hessian-vector product using a 2nd
    order adjoint formulation.

    .. note::

        Insert inexact-Hessian paper reference here

    Attributes
    ----------
    product_fact : float
        Solution tolerance for 2nd order adjoints.
    lamb : float
        ???
    scale : float
        ???
    quasi_newton : QuasiNewtonApproximation -like
        QN Hessian object to be used as preconditioner.
    """
    def __init__(self, vector_factories, optns={}):
        super(ReducedHessian, self).__init__(vector_factories, optns)

        # read reduced options
        self.product_fac = get_opt(optns, 0.001, 'product_fac')
        self.lamb = get_opt(optns, 0.0, 'lambda')
        self.scale = get_opt(optns, 1.0, 'scale')
        self.nu = get_opt(optns, 0.95, 'nu')
        self.dynamic_tol = get_opt(optns, False, 'dynamic_tol')

        # preconditioner and solver settings
        self.precond = get_opt(optns, None, 'precond')
        self.quasi_newton = None
        self.krylov = None

        # reset the linearization flag
        self._allocated = False

        # get references to individual factories
        self.primal_factory = None
        self.state_factory = None
        for factory in self.vec_fac:
            if factory._vec_type is PrimalVector:
                self.primal_factory = factory
            elif factory._vec_type is StateVector:
                self.state_factory = factory

        # request vector memory for future allocation
        self.primal_factory.request_num_vectors(4)
        self.state_factory.request_num_vectors(7)

        # initialize abtract jacobians
        self.dRdX = dRdX()
        self.dRdU = dRdU()

    def set_krylov_solver(self, krylov_solver):
        if isinstance(krylov_solver, KrylovSolver):
            self.krylov = krylov_solver
        else:
            raise TypeError('Solver is not a valid KrylovSolver')

    def set_quasi_newton(self, quasi_newton):
        if isinstance(quasi_newton, QuasiNewtonApprox):
            self.quasi_newton = quasi_newton
        else:
            raise TypeError('Object is not a valid QuasiNewtonApprox')

    def linearize(self, at_design, at_state, at_adjoint):
        """
        An abstracted "linearization" method for the matrix.

        This method does not actually factor any real matrices. It also does
        not perform expensive linear or non-linear solves. It is used to update
        internal vector references and perform basic calculations using only
        cheap matrix-vector products.

        Parameters
        ----------
        at_design : PrimalVector
            Design point at which the product is evaluated.
        at_state : StateVector
            State point at which the product is evaluated.
        at_dual : DualVector
            Lagrange multipliers at which the product is evaluated.
        at_adjoint : StateVector
            1st order adjoint variables at which the product is evaluated.
        """
        # store the linearization point
        self.at_design = at_design
        self.primal_norm = self.at_design.norm2
        self.at_state = at_state
        self.state_norm = self.at_state.norm2
        self.at_adjoint = at_adjoint

        # if this is the first ever linearization...
        if not self._allocated:

            # generate state vectors
            self.adjoint_res = self.state_factory.generate()
            self.w_adj = self.state_factory.generate()
            self.lambda_adj = self.state_factory.generate()
            self.state_work = []
            for i in xrange(4):
                self.state_work.append(self.state_factory.generate())

            # generate primal vectors
            self.pert_design = self.primal_factory.generate()
            self.reduced_grad = self.primal_factory.generate()
            self.primal_work = []
            for i in xrange(2):
                self.primal_work.append(self.primal_factory.generate())
            self._allocated = True

        # compute adjoint residual at the linearization
        self.adjoint_res.equals_objective_partial(self.at_design, self.at_state)
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[0])
        self.adjoint_res.plus(self.state_work[0])

        # compute reduced gradient at the linearization
        self.reduced_grad.equals_objective_partial(
            self.at_design, self.at_state)
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.T.product(self.at_adjoint, self.primal_work[0])
        self.reduced_grad.plus(self.primal_work[0])

    def product(self, in_vec, out_vec):
        """
        Matrix-vector product for the reduced KKT system.

        Parameters
        ----------
        in_vec : ReducedKKTVector
            Vector to be multiplied with the KKT matrix.
        out_vec : ReducedKKTVector
            Result of the operation.
        """
        # calculate perturbation
        epsilon_fd = calc_epsilon(self.primal_norm, in_vec.norm2)

        # perturb the design vector
        self.pert_design.equals_ax_p_by(1.0, self.at_design, epsilon_fd, in_vec)

        # compute total gradient at the perturbed design
        out_vec.equals_objective_partial(self.pert_design, self.at_state)
        self.dRdX.linearize(self.pert_design, self.at_state)
        self.dRdX.T.product(self.at_adjoint, self.primal_work[0])
        out_vec.plus(self.primal_work[0])

        # take the difference between perturbed and unperturbed gradient
        out_vec.minus(self.reduced_grad)

        # divide it by the perturbation
        out_vec.divide_by(epsilon_fd)

        # first adjoint system
        ####################################

        # build RHS
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.product(in_vec, self.state_work[0])
        self.state_work[0].times(-1.0)

        # solve the first 2nd order adjoint
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.solve(
            self.state_work[0], self.w_adj, rel_tol=self.product_fac)

        # second adjoint system
        #####################################

        # calculate total (dg/dx)^T*w using FD
        self.state_work[0].equals_objective_partial(
            self.pert_design, self.at_state)
        self.dRdU.linearize(self.pert_design, self.at_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[1])
        self.state_work[0].plus(self.state_work[1])
        self.state_work[0].minus(self.adjoint_res)
        self.state_work[0].divide_by(epsilon_fd)

        # multiply by -1 to use it as RHS
        self.state_work[0].times(-1.0)

        # perform state perturbation
        self.state_work[1].equals_ax_p_by(
            1.0, self.at_state, epsilon_fd, self.w_adj)

        # calculate total (dS/du)^T*z using FD
        self.state_work[2].equals_objective_partial(
            self.at_design, self.state_work[1])
        self.state_work[3].equals_ax_p_by(
            -1./epsilon_fd, self.state_work[2], 1./epsilon_fd, self.adjoint_res)
        self.dRdU.linearize(self.at_design, self.state_work[1])
        self.dRdU.T.product(self.at_adjoint, self.state_work[2])
        self.state_work[3].equals_ax_p_by(
            1., self.state_work[3], -1./epsilon_fd, self.state_work[2])

        # assemble RHS
        self.state_work[0].plus(self.state_work[3])

        # solve the second 2nd order adjoint
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.solve(
            self.state_work[0], self.lambda_adj, rel_tol=self.product_fac)

        # assemble the Hessian-vector product using 2nd order adjoints
        ##############################################################

        # apply lambda_adj to the design part of the jacobian
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.T.product(self.lambda_adj, self.primal_work[0])
        out_vec.plus(self.primal_work[0])

        # apply w_adj to the cross-derivative part of the jacobian
        self.primal_work[0].equals_objective_partial(
            self.at_design, self.state_work[1])
        self.dRdX.linearize(self.at_design, self.state_work[1])
        self.dRdX.T.product(self.at_adjoint, self.primal_work[1])
        self.primal_work[0].plus(self.primal_work[1])
        self.primal_work[0].equals_ax_p_by(
            1./epsilon_fd, self.primal_work[0],
            -1./epsilon_fd, self.reduced_grad)
        out_vec.plus(self.primal_work[0])

        # update quasi-Newton method if necessary
        if self.quasi_newton is not None:
            self.quasi_newton.add_correction(in_vec, out_vec)

        # add globalization if necessary
        if self.lamb > numpy.finfo(float).eps:
            out_vec.equals_ax_p_by(
                1.-self.lamb, out_vec, self.lamb*self.scale, in_vec)

    def solve(self, rhs, solution, rel_tol=None):
        """
        Solve the linear system defined by this matrix using the embedded
        krylov solver.

        Parameters
        ----------
        rhs : PrimalVector
            Right hand side vector for the system.
        solution : PrimalVector
            Solution of the system.
        rel_tol : float, optional
            Relative tolerance for the krylov solver.
        """
        # make sure we have a krylov solver
        if self.krylov is None:
            raise AttributeError('krylov solver not set')

        # define the preconditioner
        if self.precond is not None:
            if self.quasi_newton is not None:
                precond = self.quasi_newton.solve
            else:
                raise AttributeError('preconditioner is specified but not set')
        else:
            eye = IdentityMatrix()
            precond = eye.product

        # update the solution tolerance if necessary
        if isinstance(rel_tol, float):
            self.krylov.rel_tol = rel_tol

        # trigger the solution
        return self.krylov.solve(self.product, rhs, solution, precond)
