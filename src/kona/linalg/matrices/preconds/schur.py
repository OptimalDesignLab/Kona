import numpy as np
from kona.linalg.matrices.hessian.basic import BaseHessian

class ApproxSchur(BaseHessian):
    """
    Uses an approximate SVD to approximate the Schur complement of the primal-dual matrix.

    Parameters
    ----------


    Attributes
    ----------
    subspace_size : int
        Number of iterations in the Lanczos algorithm, and the number of singular values
        approximated by the decomposition.
    """
    def __init__(self,  vector_factories, optns=None):
        super(ApproxSchur, self).__init__(vector_factories, optns)

        # request vectors needed (work)
        dual_work = None
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(1)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(1)

        # initialize the total constraint jacobian block
        self.dCdX = TotalConstraintJacobian(vector_factories)

        # Questions for Alp:
        # 1) LowRankSVD needs a rev_factory for rectangular matrices; what if CompositeDualVector?
        # 2) Would it be safe to implement a CompositeDualVector factory?
        # 3) Looks like LowRankSVD assumes BaseVectors, correct?

        # initialize the low-rank SVD
        # Will these definitions persist!!!!
        def fwd_mat_vec(in_vec, out_vec):
            self.dCdX.product(in_vec, out_vec)
        def rev_mat_vec(in_vec, out_vec):
            self.dCdX.T.product(in_vec, out_vec)
        self.Schur_inv = LowRankSVD(fwd_mat_vec, self.primal_factory,
                                    rev_mat_vec, [self.eq_factory, self.ineq_factory])

    def linearize(self, at_primal, at_state, scale=1.0):
        # need to generate dual_work here

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
            self.dCdX.product(in_primal, dual_work)
            dual_work.plus(in_dual)
            # Step 2: v_d <-- (A*A^T)^-1*w
            self.Schur_inv.inv_schur_product(dual_work, out_dual)
            # Step 3: v_p <-- u_p - A^T*v_d
            self.dCdX.T.product(out_dual, out_primal)
        else:
            # no preconditioning for unconstrained case
            out_primal.equals(0.)
        out_primal.equals_ax_p_by(1., in_primal, -1., out_primal)
