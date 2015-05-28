import sys, gc
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

        Insert paper reference here

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
        self.scale = get_opt(optns, 0.0, 'scale')
        self.nu = get_opt(optns, 0.95, 'nu')
        self.dynamic_tol = get_opt(optns, False, 'dynamic_tol')

        # preconditioner and solver settings
        self.precond = get_opt(optns, None, 'precond')
        self.quasi_newton = None
        self.krylov = None

        # reset the linearization flag
        self._linearized = False

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

        # store the linearization point
        self.at_design = at_design
        self.primal_norm = self.at_design.norm2
        self.at_state = at_state
        self.state_norm = self.at_state.norm2
        self.at_adjoint = at_adjoint

        # if this is the first ever linearization...
        if not self._linearized:

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
            self._linearized = True

        # compute adjoint residual at the linearization
        self.adjoint_res.equals_objective_partial(self.at_design, self.at_state)
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[0])
        self.adjoint_res.plus(self.state_work[0])

        # compute reduced gradient at the linearization
        self.reduced_grad.equals_objective_partial(self.at_design, self.at_state)
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.T.product(self.at_adjoint, self.primal_work[0])
        self.reduced_grad.plus(self.primal_work[0])

    def product(self, in_vec, out_vec):

        # perturb the design vector
        epsilon_fd = calc_epsilon(self.primal_norm, in_vec.norm2)
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

        # build RHS for first adjoint system
        ####################################

        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.product(in_vec, self.state_work[0])
        self.state_work[0].times(-1.0)

        # solve the first 2nd order adjoint
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.solve(self.state_work[0], self.product_fac, self.w_adj)

        # build RHS for second adjoint system
        #####################################

        # calculate 1st order adjoint residual at perturbed design
        self.state_work[0].equals_objective_partial(self.pert_design, self.at_state)
        self.dRdU.linearize(self.pert_design, self.at_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[1])
        self.state_work[0].plus(self.state_work[1])

        # take the difference between perturbed and unperturbed residuals
        self.state_work[0].minus(self.adjoint_res)

        # divide it by the perturbation
        self.state_work[0].divide_by(epsilon_fd)

        # multiply by -1 to use it as RHS
        self.state_work[0].times(-1.0)

        # perform state perturbation
        epsilon_fd = calc_epsilon(self.state_norm, self.w_adj.norm2)
        eps_r = 1./epsilon_fd
        self.state_work[1].equals_ax_p_by(1.0, self.at_state, epsilon_fd, self.w_adj)

        # calculate 1st order adjoint residual at perturbed state,
        # take difference and divide by perturbation
        self.state_work[2].equals_objective_partial(self.at_design, self.state_work[1])
        self.state_work[3].equals_ax_p_by(-eps_r, self.state_work[2], eps_r, self.adjoint_res)
        self.dRdU.linearize(self.at_design, self.state_work[1])
        self.dRdU.T.product(self.at_adjoint, self.state_work[2])
        self.state_work[3].equals_ax_p_by(1.0, self.state_work[3], -eps_r, self.state_work[2])

        # assemble the second adjoint RHS
        self.state_work[0].plus(self.state_work[3])

        # solve the second 2nd order adjoint
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.solve(self.state_work[0], self.product_fac, self.lambda_adj)

        # assemble the Hessian-vector product using 2nd order adjoints
        ##############################################################

        # apply lambda_adj to the design part of the jacobian
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.T.product(self.lambda_adj, self.primal_work[0])
        out_vec.plus(self.primal_work[0])

        # apply w_adj to the cross-derivative part of the jacobian
        self.primal_work[0].equals_objective_partial(self.at_design, self.state_work[0])
        self.dRdX.linearize(self.at_design, self.state_work[0])
        self.dRdX.T.product(self.at_adjoint, self.primal_work[1])
        self.primal_work[0].plus(self.primal_work[1])
        self.primal_work[0].equals_ax_p_by(eps_r, self.primal_work[0], -eps_r, self.reduced_grad)
        out_vec.plus(self.primal_work[0])

        # update quasi-Newton method if necessary
        if self.quasi_newton is not None:
            self.quasi_newton.add_correction(in_vec, out_vec)

        # add globalization if necessary
        if self.lamb > numpy.finfo(float).eps:
            out_vec.equals_ax_p_by((1.-self.lamb), out_vec, self.lamb*self.scale, in_vec)

    def solve(self, in_vec, out_vec, rel_tol=None):

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
        return self.krylov.solve(self.product, in_vec, out_vec, precond)
