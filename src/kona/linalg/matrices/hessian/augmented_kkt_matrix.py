
from kona.options import get_opt
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.preconds import NestedNormalStepPreconditioner
from kona.linalg.solvers.krylov import GCROT, FGMRES

class AugmentedKKTMatrix(BaseHessian):
    """
    Matrix object for the the normal system associated with the reduced KKT
    system.

    The normal system is defined as:

    .. math::
        \\begin{bmatrix}
        I && 0 && A^T \\\\
        0 && I && -\\Sigma \\\\
        A && -\\Sigma && 0
        \\end{bmatrix}

    This matrix is used to solve the normal-step in a composite-step algorithm.
    """
    def __init__(self, vector_factories, optns={}):
        super(AugmentedKKTMatrix, self).__init__(vector_factories, optns)

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
        self.dual_factory.request_num_vectors(1)
        self._allocated = False

        # decide which krylov solver we use
        self.use_gcrot = get_opt(optns, True, 'use_gcrot')

        # initialize the constraint jacobian
        self.A = TotalConstraintJacobian(vector_factories)

        # get preconditioner options
        self.precond = get_opt(optns, None, 'precond')
        if self.precond is None:
            self.nested_svd = None
            eye = IdentityMatrix()
            self.precond = eye.product
        elif self.precond == 'nested_svd':
            svd_optns = {
                'lanczos_size' : get_opt(optns, 10, 'lanczos_size'),
            }
            self.nested_svd = NestedNormalStepPreconditioner(
                self.primal_factory, self.dual_factory, self.A, svd_optns)
            self.precond = self.nested_svd.solve
        else:
            raise TypeError('Invalid preconditioner!' +
                            'Can be either \'nested_svd\' or None.')

        # initialize the internal krylov solver
        if self.use_gcrot:
            krylov_optns = {
                'out_file' : get_opt(
                    optns, 'kona_normal_gcrot.dat', 'out_file'),
                'subspace_size' : get_opt(optns, 10, 'subspace_size'),
                'max_recycle' : get_opt(optns, 10, 'max_recycle'),
                'max_outer' : get_opt(optns, 10, 'max_outer'),
                'max_matvec' : get_opt(optns, 50, 'max_matvec'),
                'check_res' : get_opt(optns, True, 'check_res'),
                'rel_tol' : get_opt(optns, 1e-3, 'rel_tol'),
                'abs_tol' : get_opt(optns, 1e-5, 'abs_tol')
            }
            self.krylov = GCROT(
                self.primal_factory,
                optns=krylov_optns,
                dual_factory=self.dual_factory)
        else:
            krylov_optns = {
                'out_file' : get_opt(
                    optns, 'kona_normal_fgmres.dat', 'out_file'),
                'subspace_size' : get_opt(optns, 10, 'subspace_size'),
                'check_res' : get_opt(optns, True, 'check_res'),
                'rel_tol' : get_opt(optns, 1e-3, 'rel_tol'),
                'abs_tol' : get_opt(optns, 1e-5, 'abs_tol')
            }
            self.krylov = FGMRES(
                self.primal_factory,
                optns=krylov_optns,
                dual_factory=self.dual_factory)

    def linearize(self, at_kkt, at_state):
        # store references to the evaluation point
        self.at_design = at_kkt._primal._design
        self.at_slack = at_kkt._primal._slack
        self.at_dual = at_kkt._dual
        self.at_state = at_state

        # generate a work vector
        if not self._allocated:
            self.slack_term = self.dual_factory.generate()
            self._allocated = True
        self.slack_term.exp(self.at_slack)
        self.slack_term.restrict()

        # linearize the constraint jacobian
        self.A.linearize(self.at_design, self.at_state)

        # linearize the preconditioner
        if self.nested_svd is not None:
            self.nested_svd.linearize(self.at_dual, self.at_slack)

        # reset the krylov subspace
        if self.use_gcrot:
            self.krylov.clear_subspace()

        # do aliasing on a work vector
        self.dual_work = self.A.dual_work

    def product(self, in_vec, out_vec):
        # compute the design product
        # out_design = in_design + A^T*in_dual
        self.A.T.product(in_vec._dual, out_vec._primal._design)
        out_vec._primal._design.plus(in_vec._primal._design)

        # compute the slack product
        # out_slack = in_slack - Sigma*in_dual
        out_vec._primal._slack.equals(in_vec._dual)
        out_vec._primal._slack.times(self.slack_term)
        out_vec._primal._slack.times(-1.)
        out_vec._primal._slack.plus(in_vec._primal._slack)
        out_vec._primal._slack.restrict()

        # compute the dual product
        # out_dual = A*in_design - Sigma*in_slack
        self.A.product(in_vec._primal._design, out_vec._dual)
        self.dual_work.equals(in_vec._primal._slack)
        self.dual_work.times(self.slack_term)
        self.dual_work.times(-1.)
        out_vec._dual.plus(self.dual_work)

    def solve(self, rhs, solution, rel_tol=None):
        # set krylov relative tolerance
        if rel_tol is not None:
            tmp_rel_tol = self.krylov.rel_tol
            self.krylov.rel_tol = rel_tol

        # solve the system
        solution.equals(0.0)
        self.krylov.solve(self.product, rhs, solution, self.precond)

        # reset the tolerance for the krylov object
        if rel_tol is not None:
            self.krylov.rel_tol = tmp_rel_tol
