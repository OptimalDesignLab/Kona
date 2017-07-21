import numpy as np
from kona.linalg.vectors.composite import CompositeFactory, PrimalDualVector, CompositeDualVector
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.preconds.low_rank_svd import LowRankSVD

class ApproxSchur(BaseHessian):
    """
    Uses an approximate SVD to estimate the inverse Schur complement of the primal-dual matrix.
    Specifically, this preconditioner forms an approximate inverse of the augmented matrix

    .. math::
        \\begin{bmatrix} I && -A^T \\\\ -A && 0 \\end{bmatrix}

    where :math:`A` is the total constraint Jacobian.  The preconditioner is based on an LU
    decomposition with the Schur complement approximated by a low-rank update to a diagonal matrix.
    The action of the preconditioner on a vector :math:`[u_p^T u_v^T]^T` is given by

    .. math::
        \\begin{bmatrix} u_p - \\tilde{A}^T (\\tilde{A}\\tilde{A}^T)^{-1}(\\tilde{A}u_p - u_d) \\\\
        - (\\tilde{A}\\tilde{A}^T)^{-1}(\\tilde{A}u_p - u_d) \\end{bmatrix}

    where :math:`\\tilde{A}` is the low-rank approximation to the constraint Jacobian, and
    :math:`(\\tilde{A}\\tilde{A}^{T})^{-1}` is found using a pseudo-inverse-like approximation: see
    LowRankSVD.inv_schur_prod for further details on this.

    Attributes
    ----------
    dCdX : TotalConstraintJacobian
        Used in Lanczos bi-diagonalization to find low-rank SVD
    dCdX_approx : LowRankSVD
        Used to approximate the inverse of (dCdX)*(dCdX^T)
    """
    def __init__(self, vector_factories, optns=None):
        super(ApproxSchur, self).__init__(vector_factories, optns)

        # set-up the dual factory
        self.dual_factory = None
        if self.eq_factory is not None and self.ineq_factory is not None:
            self.dual_factory = CompositeFactory(self.eq_factory._memory, CompositeDualVector)
        elif self.eq_factory is not None:
            self.dual_factory = self.eq_factory
        else:
            self.dual_factory = self.ineq_factory

        # request vectors needed (dual_work)
        self.dual_work = None
        if self.dual_factory is not None:
            self.dual_factory.request_num_vectors(1)

            # initialize the total constraint jacobian block
            self.dCdX = TotalConstraintJacobian(vector_factories)

            # initialize the low-rank SVD
            def fwd_mat_vec(in_vec, out_vec):
                self.dCdX.product(in_vec, out_vec)

            def rev_mat_vec(in_vec, out_vec):
                self.dCdX.T.product(in_vec, out_vec)

            self.dCdX_approx = LowRankSVD(fwd_mat_vec, self.primal_factory,
                                          rev_mat_vec, self.dual_factory, optns)

        else:
            # if self.dual_factory is None, there are no constraints
            self.dCdX = None
            self.dCdX_approx = None

        self._allocated = False

    def linearize(self, at_primal, at_state, scale=1.0):
        # store references to the evaluation point, and save the scaling of constraint terms
        self.at_design = at_primal
        self.at_state = at_state
        self.scale = scale

        # get the total constraint Jacobian ready, and then use it in Lanczos bi-diagonalization
        if self.dual_factory is not None:
            self.dCdX.linearize(self.at_design, self.at_state, scale=self.scale)
            self.dCdX_approx.linearize()

        # check if dual_work needs to be allocated
        if not self._allocated:
            if self.dual_factory is not None:
                self.dual_work = self.dual_factory.generate()
            self._allocated = True

    def product(self, in_vec, out_vec):
        assert isinstance(in_vec, PrimalDualVector), \
            "ApproxSchur() >> in_vec must be a PrimalDualVector!"
        assert isinstance(out_vec, PrimalDualVector), \
            "ApproxSchur() >> out_vec must be a PrimalDualVector!"
        # set some aliases
        dual_work = self.dual_work
        in_primal = in_vec.primal
        in_dual = in_vec.get_dual()
        out_primal = out_vec.primal
        out_dual = out_vec.get_dual()
        if dual_work is not None:
            # Step 1: w <-- A*u_p + u_d
            self.dCdX_approx.approx_fwd_prod(in_primal, dual_work)
            dual_work.plus(in_dual)
            # Step 2: v_d <-- -(A*A^T)^-1*w
            self.dCdX_approx.inv_schur_prod(dual_work, out_dual)
            out_dual.times(-1.)
            # Step 3: v_p <-- A^T*v_d
            self.dCdX_approx.approx_rev_prod(out_dual, out_primal)
        else:
            # no preconditioning for unconstrained case
            out_primal.equals(0.)
        # v_p <-- v_p + u_p
        out_primal.equals_ax_p_by(1., in_primal, 1., out_primal)
