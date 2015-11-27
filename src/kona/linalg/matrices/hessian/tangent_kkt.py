
from kona.options import get_opt
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.matrices.hessian import ConstrainedHessian, NormalKKTMatrix
from kona.linalg.solvers.krylov import STCG

class TangentKKTMatrix(BaseHessian):
    """
    Matrix object for the the tangent system associated with the reduced KKT
    system.

    The tangent system is defined as:

    .. math::
        \\begin{bmatrix}
        W && 0 \\\\
        0 && -\\Sigma \\Lambda \\\\
        \\end{bmatrix}

    This matrix is used to solve the tangent-step in a composite-step algorithm.
    """
    def __init__(self, vector_factories, optns={}):
        super(TangentKKTMatrix, self).__init__(vector_factories, optns)

        # get references to individual factories
        self.primal_factory = None
        self.state_factory = None
        self.dual_factory = None
        for factory in self.vec_fac:
            if factory._vec_type is PrimalVector:
                self.primal_factory = factory
            elif factory._vec_type is StateVector:
                self.state_factory = factory
            elif factory._vec_type is DualVector:
                self.dual_factory = factory
            else:
                raise TypeError('Invalid vector factory!')
        self.dual_factory.request_num_vectors(3)
        self._allocated = False

        # null-space projector
        self.proj_cg = None

        # trust radius settings
        self.radius = None
        self.trust_active = False
        self.pred = 0.

        # initialize the Hessian block
        self.W = ConstrainedHessian(vector_factories)

        # initialize the internal krylov method
        krylov_optns = {
            'out_file' : get_opt(optns, 'kona_tangent_stcg.dat', 'out_file'),
            'subspace_size' : get_opt(optns, 20, 'subspace_size'),
            'proj_cg'  : True,
            'check_res' : get_opt(optns, True, 'check_res'),
            'rel_tol'  : get_opt(optns, 1e-3, 'rel_tol'),
        }
        self.krylov = STCG(
            self.primal_factory,
            optns=krylov_optns,
            dual_factory=self.dual_factory)

    def set_projector(self, proj_cg):
        if isinstance(proj_cg, NormalKKTMatrix):
            self.proj_cg = proj_cg
        else:
            raise TypeError('Invalid null-space projector!')

    def linearize(self, at_kkt, at_state, at_adjoint):
        # store references to the evaluation point
        self.at_design = at_kkt._primal._design
        self.at_slack = at_kkt._primal._slack
        self.at_dual = at_kkt._dual
        self.at_state = at_state
        self.at_adjoint = at_adjoint

        if not self._allocated:
            self.dual_in = self.dual_factory.generate()
            self.dual_out = self.dual_factory.generate()
            self.slack_term = self.dual_factory.generate()
            self._allocated = True
        self.slack_term.exp(self.at_slack)
        self.slack_term.restrict()

        # linearize the constraint jacobian and constrained hessian
        self.W.linearize(
            self.at_design, self.at_dual, self.at_state, self.at_adjoint)

    def product(self, in_vec, out_vec):
        # compute the design component
        # out_design = W*in_design
        self.W.product(in_vec._design, out_vec._design)
        # compute the slack component
        # out_slack = - Sigma * Lambda * in_slack
        out_vec._slack.equals(in_vec._slack)
        out_vec._slack.times(self.at_dual)
        out_vec._slack.times(self.slack_term)
        out_vec._slack.times(-1.)

    def precond(self, in_vec, out_vec):
        in_kkt = ReducedKKTVector(in_vec, self.dual_in)
        in_kkt._dual.equals(0.0)
        out_kkt = ReducedKKTVector(out_vec, self.dual_out)
        self.proj_cg.solve(in_kkt, out_kkt, rel_tol=1e-6)

    def solve(self, rhs, solution, rel_tol=None):
        if self.radius is None:
            raise ValueError('Trust radius not set!')

        if self.proj_cg is None:
            raise ValueError('CG projection preconditioner not set!')

        # set krylov relative tolerance and radius
        self.krylov.radius = self.radius
        if rel_tol is None:
            self.krylov.rel_tol = self.rel_tol
        else:
            self.krylov.rel_tol = rel_tol

        # solve the system
        solution.equals(0.0)
        self.pred, self.trust_active = \
            self.krylov.solve(self.product, rhs, solution, self.precond)
