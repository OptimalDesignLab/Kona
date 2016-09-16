from kona.linalg.matrices.hessian.basic import BaseHessian

class QuasiNewtonKKTMatrix(BaseHessian):
    """
    An approximation of the KKT matrix using a limited memory SR1 for the
    Lagrangian Hessian, and a 2nd order adjoint formulation for the constraint
    jacobian.
    """
    def __init__(self, vector_factories, optns={}):

        super(QuasiNewtonKKTMatrix, self).__init__(vector_factories, optns)

        # read reduced options
        self.product_fac = get_opt(optns, 0.001, 'product_fac')
        self.product_tol = 1.0
        self.lamb = get_opt(optns, 0.0, 'lambda')
        self.scale = get_opt(optns, 1.0, 'scale')
        self.grad_scale = get_opt(optns, 1.0, 'grad_scale')
        self.feas_scale = get_opt(optns, 1.0, 'feas_scale')
        self.dynamic_tol = get_opt(optns, False, 'dynamic_tol')
        max_stored = get_opt(optns, 10, 'max_stored')

        # get references to individual factories
        self.primal_factory = None
        self.state_factory = None
        self.eq_factory = None
        self.ineq_factory = None
        for factory in self.vec_fac:
            if factory._vec_type is DesignVector:
                self.primal_factory = factory
            elif factory._vec_type is StateVector:
                self.state_factory = factory
            elif factory._vec_type is DualVectorEQ:
                self.eq_factory = factory
            elif factory._vec_type is DualVectorINEQ:
                self.ineq_factory = factory

        # set empty solver handle
        self.krylov = None

        # reset the linearization flag
        self._allocated = False

        # initialize the L-SR1 QN object
        optns = {'max_stored' : max_stored}
        self.quasi_newton = LimitedMemorySR1(self.primal_factory, optns)

        # initialize the constraint jacobian objects
        self.cnstr_jac = TotalConstraintJacobian(self.vec_fac)

        # request vector memory for future allocation
        self.primal_factory.request_num_vectors(3)
        self.state_factory.request_num_vectors(6)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(3)

    def set_krylov_solver(self, krylov_solver):
        if isinstance(krylov_solver, KrylovSolver):
            self.krylov = krylov_solver
        else:
            raise TypeError('Solver is not a valid KrylovSolver')

    def linearize(self, at_kkt, at_state):
        # if this is the first ever linearization...
        if not self._allocated:
            # generate primal vectors
            self.primal_work = self.primal_factory.generate()
            # generate dual vectors
            self.slack_block = None
            if self.ineq_factory is not None:
                self.slack_block= self.ineq_factory.generate()

            self._allocated = True

        # store the linearization point
        if isinstance(at_kkt.primal, CompositePrimalVector):
            self.at_design = at_kkt.primal.design
            self.at_slack = at_kkt.primal.slack
            if isinstance(at_kkt.dual, CompositeDualVector):
                self.at_dual_ineq = at_kkt.dual.ineq
            else:
                self.at_dual_ineq = at_kkt.dual
        else:
            self.at_design = at_kkt.primal
            self.at_slack = None
            self.at_dual_ineq = None
        self.at_state = at_state

        # pre compute the slack block
        self.slack_block.equals(self.at_slack)
        self.slack_block.pow(-1.)
        self.slack_block.times(self.at_dual_ineq)

        # linearize the constraint jacobian
        self.cnstr_jac.linearize(self.at_design, self.at_state)

    def add_correction(self, delta_x, grad_diff):
        self.quasi_newton.add_correction(delta_x, grad_diff)

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
        # do some aliasing for input vector components
        if self.at_slack is not None:
            in_design = in_vec.primal.design
            out_design = out_vec.primal.design
            in_slack = in_vec.primal.slack
            out_slack = out_vec.primal.slack
        else:
            in_design = in_vec.primal
            out_design = out_vec.primal
            in_slack = None

        # start with the quasi newton part of the product
        self.quasi_newton.product(in_design, out_design)

        # get the transposed cnstr jac product
        self.cnstr_jac.T.product(in_vec.dual, self.primal_work)
        out_design.plus(self.primal_work)

        # get the forward cnstr jac product
        self.cnstr_jac.product(in_design, out_vec.dual)

        # now take care of the slack block
        if in_slack is not None:
            out_slack.equals(in_slack)
            out_slack.times(self.slack_block)

# imports here to prevent circular errors
from kona.options import get_opt
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.hessian import LimitedMemorySR1
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.solvers.krylov.basic import KrylovSolver