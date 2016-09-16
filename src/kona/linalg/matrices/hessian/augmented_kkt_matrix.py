from kona.linalg.matrices.hessian.basic import BaseHessian

class AugmentedKKTMatrix(BaseHessian):
    """
    Matrix object for the the normal system associated with the reduced KKT
    system.

    The normal system is defined as:

    .. math::
        \\begin{bmatrix}
        I && 0 && Aeq^T && Ain^T \\\\
        0 && Sigma && 0 && I \\\\
        Aeq && 0 && 0 && 0 \\\\
        Ain && I && 0 && 0
        \\end{bmatrix}

    This matrix is used to solve the normal-step in a composite-step algorithm.
    """
    def __init__(self, vector_factories, optns={}):
        super(AugmentedKKTMatrix, self).__init__(vector_factories, optns)

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
                self.eq_factory = factory
            else:
                raise TypeError('Invalid vector factory!')

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
                eq_factory=self.eq_factory,
                ineq_factory=self.ineq_factory)
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
                eq_factory=self.eq_factory,
                ineq_factory=self.ineq_factory)

    def linearize(self, at_kkt, at_state):
        # store references to the evaluation point
        if isinstance(at_kkt.primal, CompositePrimalVector):
            self.at_design = at_kkt.primal.design
            self.at_slack = at_kkt.primal.slack
        else:
            self.at_design = at_kkt.primal
            self.at_slack = None
        self.at_dual = at_kkt.dual
        self.at_state = at_state

        # linearize the constraint jacobian
        self.A.linearize(self.at_design, self.at_state)

        # reset the krylov subspace
        if self.use_gcrot:
            self.krylov.clear_subspace()

        # do aliasing on a work vector
        self.dual_work = self.A.dual_work

    def product(self, in_vec, out_vec):
        # do some aliasing for the vectors
        if isinstance(in_vec.primal, CompositePrimalVector):
            in_design = in_vec.primal.design
            out_design = out_vec.primal.slack
            in_slack = in_vec.primal.slack
            out_slack = out_vec.primal.slack
            if isinstance(in_vec.dual, CompositeDualVector):
                in_dual_ineq = in_vec.dual.ineq
                out_dual_ineq = out_vec.dual.ineq
            else:
                in_dual_ineq = in_vec.dual
                out_dual_ineq = out_vec.dual
        else:
            in_design = in_vec.primal
            out_design = out_vec.primal
            in_slack = None

        # compute the design product
        # out_design = in_design + A^T*in_dual
        self.A.T.product(in_vec.dual, out_design)
        out_design.plus(in_design)

        # compute the dual product
        # out_dual = A*in_design
        self.A.product(in_design, out_vec.dual)

        # deal with slack terms
        if in_slack is not None:
            # compute the slack product
            # out_slack = I * in_dual_ineq
            out_slack.equals(in_dual_ineq)
            # now modify the dual product
            # out_dual_ineq += I * in_slack
            out_dual_ineq.plus(in_slack)

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

# imports here to prevent circular errors
from kona.options import get_opt
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.solvers.krylov import GCROT, FGMRES