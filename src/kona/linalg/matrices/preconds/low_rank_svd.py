import numpy as np

from kona.options import get_opt
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.solvers.util import lanczos_bidiag

class LowRankSVD(BaseHessian):
    """
    This object preconditions the normal KKT system by solving a modified
    version of this system via a low-rank singular-value decomposition.

    The normal KKT system is given as:

    .. math::
        \\begin{bmatrix}
        \\mathsf{I} && 0 && \\mathsf{A}^T \\\\
        0 && \\mathsf{I} && -\\Sigma \\Lambda \\\\
        \\mathsf{A} && -\\Sigma && 0
        \\end{bmatrix}
        \\begin{bmatrix}
        v_d \\\\ v_s \\\\ v_{\\lambda}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        u_d \\\\ u_s \\\\ u_{\\Lambda}
        \\end{bmatrix}

    Here the :math:`\\mathsf{A}` matrix is the total constraint jacobian,
    :math:`(u_d, u_s, u_{\\lambda})^T` is the vector to be preconditioned, and
    :math:`(v_d, v_s, v_{\\lambda})^T` is the preconditioned vector.
    :math:`\\Sigma` is a diagonal matrix of slack terms, defined as
    :math:`\\Sigma = diag(e^s)`.

    First we take the slacks out of the system via
    :math:`v_s = u_s + \\Sigma b_{\\lambda}`.

    .. math::
        \\begin{bmatrix}
        \\mathsf{I} && \\mathsf{A}^T \\\\
        \\mathsf{A} && -\\Sigma^2
        \\end{bmatrix}
        \\begin{bmatrix}
        v_d \\\\ v_{\\lambda}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        u_d \\\\ u_d + \\Sigma u_s
        \\end{bmatrix}

    Then we perform a substitution
    :math:`\\hat{v}_{\\lambda} = \\Lambda^{-1} v_{\\lambda}` where
    :math:`\\Lambda = diag(\\lambda)`.

    .. math::
        \\begin{bmatrix}
        \\mathsf{I} && \\mathsf{A}^T \\Lambda \\\\
        \\mathsf{A} && -\\Sigma^2 \\Lambda
        \\end{bmatrix}
        \\begin{bmatrix}
        v_d \\\\ \\hat{v}_{\\lambda}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        u_d \\\\ u_d + \\Sigma u_s
        \\end{bmatrix}

    Now we restablish the symmetry of the system by multiplying the dual
    equation (2nd row) by :math:`\\Lambda`:

    .. math::
        \\begin{bmatrix}
        \\mathsf{I} && \\mathsf{A}^T \\Lambda \\\\
        \\Lambda \\mathsf{A} && -(\\Sigma \\Lambda)^2
        \\end{bmatrix}
        \\begin{bmatrix}
        v_d \\\\ \\hat{v}_{\\lambda}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        u_d \\\\ \\Lambda(u_d + \\Sigma u_s)
        \\end{bmatrix}

    The Lanczos bi-diagonalization is applied to the
    :math:`\\Lambda \\mathsf{A}` to produce
    :math:`\\Lambda \\mathsf{A} = \\mathsf{U}\\mathsf{S}\\mathsf{V}^T`.

    The modified system above is then solved recognizing that
    :math:`\\Sigma \\Lambda \\rightarrow 0` at convergence.

    1. Compute :math:`\\hat{v}_{\\lambda} =
        \\mathsf{U} \\mathsf{S}^{-1} \\mathsf{U}^T(\\Lambda(u_d +
        \\Sigma u_s) - \\mathsf{U}\\mathsf{S}\\mathsf{V}^Tu_p)`.
    2. Compute :math:`v_p = u_p -
        \\mathsf{V}\\mathsf{S}\\mathsf{U}^T\\hat{v}_{\\lambda}`.
    3. Compute :math:`v_s = u_s + \\Sigma \\Lambda \\hat{v}_{\\lambda}`.
    4. Compute :math:`v_{\\lambda} = \\Lambda \\hat{v}_{\\lambda}`.
    """

    def __init__(self, vector_factories, optns):
        super(LowRankSVD, self).__init__(vector_factories, optns)

        # set basic internal options
        self.use_precond = True
        self.subspace_size = get_opt(optns, 10, 'lanczos_size')

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

        # set the constraint jacobian matrix
        if isinstance(A, TotalConstraintJacobian):
            self.A = A
        else:
            raise TypeError('Invalid constraint jacobian!')

        # reset the linearization flag
        self._allocated = False

        # request vector memory for future allocation
        self.primal_factory.request_num_vectors(5 + 2*self.subspace_size)
        self.state_factory.request_num_vectors(6)
        self.dual_factory.request_num_vectors(3 + 2*self.subspace_size)

    def linearize(self, at_dual, at_slack):

        if not self._allocated:
            # this is the first allocation
            # generate subspace vectors
            self.q_work = self.primal_factory.generate()
            self.p_work = self.dual_factory.generate()
            self.Q = []
            self.V = []
            self.P = []
            self.U = []
            for i in xrange(self.subspace_size):
                self.Q.append(self.primal_factory.generate())
                self.V.append(self.primal_factory.generate())
                self.P.append(self.dual_factory.generate())
                self.U.append(self.dual_factory.generate())
            self.Q.append(self.primal_factory.generate())
            # generate work vectors for the product
            self.design_work = self.primal_factory.generate()
            self.dual_work = self.dual_factory.generate()
            self.slack_term = self.dual_factory.generate()
            self.state_work = self.state_factory.generate()
            # flip the allocation flag
            self._allocated = True

        # do some aliasing
        self.at_dual = at_dual
        self.at_slack = at_slack
        self.slack_term.exp(self.at_slack)
        self.slack_term.restrict()

        def _fwd_mat_vec(in_vec, out_vec):
            self.A.product(in_vec, out_vec)
            out_vec.times(self.at_dual)

        def _rev_mat_vec(in_vec, out_vec):
            self.dual_work.equals(in_vec)
            self.dual_work.times(self.at_dual)
            self.A.T.product(self.dual_work, out_vec)

        # bi-diagonalize the constraint jacobian blocks
        B = lanczos_bidiag(_fwd_mat_vec, self.Q, self.q_work,
                           _rev_mat_vec, self.P, self.p_work)

        # decompose the bi-diagonal matrix
        u_tmp, s_tmp, vT_tmp = np.linalg.svd(B, full_matrices=0)
        v_tmp = vT_tmp.T

        # save the singular values
        self.S = np.diag(s_tmp)

        # diagonalize the singular values and calculate inverse
        self.S2_inv = np.linalg.inv(self.S**2)

        # calculate V = Q*v_tmp
        for j in xrange(len(self.V)):
            self.V[j].equals(0.0)
            for i in xrange(len(v_tmp[:, j])):
                self.q_work.equals(self.Q[i])
                self.q_work.times(v_tmp[i, j])
                self.V[j].plus(self.q_work)

        # calculate U = P*u_tmp
        for j in xrange(len(self.U)):
            self.U[j].equals(0.0)
            for i in xrange(len(u_tmp[:, j])):
                self.p_work.equals(self.P[i])
                self.p_work.times(u_tmp[i, j])
                self.U[j].plus(self.p_work)

    def mult_lambda_A(self, in_vec, out_vec):
        VT_in = np.zeros(len(self.V))
        for i in xrange(len(self.V)):
            VT_in[i] = self.V[i].inner(in_vec)
        SVT_in = np.dot(self.S, VT_in)
        out_vec.equals(0.0)
        for i in xrange(len(self.U)):
            out_vec.equals_ax_p_by(1., out_vec, SVT_in[i], self.U[i])

    def mult_A_T_lambda(self, in_vec, out_vec):
        UT_vec = np.zeros(len(self.U))
        for i in xrange(len(self.U)):
            UT_vec[i] = self.U[i].inner(in_vec)
        SUT_vec = np.dot(self.S, UT_vec)
        out_vec.equals(0.0)
        for i in xrange(len(self.V)):
            out_vec.equals_ax_p_by(1., out_vec, SUT_vec[i], self.V[i])

    def mult_lambda_AAT_lambda_inv(self, in_vec, out_vec):
        UT_vec = np.zeros(len(self.U))
        for i in xrange(len(self.U)):
            UT_vec[i] = self.U[i].inner(in_vec)
        S2_inv_UT_vec = np.dot(self.S2_inv, UT_vec)
        out_vec.equals(0.0)
        for i in xrange(len(self.U)):
            out_vec.equals_ax_p_by(1., out_vec, S2_inv_UT_vec[i], self.U[i])

    def product(self, in_vec, out_vec):
        # do some aliasing
        in_design = in_vec._primal._design
        in_slack = in_vec._primal._slack
        in_dual = in_vec._dual
        out_design = out_vec._primal._design
        out_slack = out_vec._primal._slack
        out_dual = out_vec._dual

        # compute the modified dual term
        # US^{-2}U^T*(Lambda*(in_dual + Sigma*in_slack) - USV^T*in_design)
        self.mult_lambda_A(in_design, out_dual)
        out_dual.times(-1.)
        self.dual_work.equals(in_slack)
        self.dual_work.times(self.slack_term)
        self.dual_work.plus(in_dual)
        self.dual_work.times(self.at_dual)
        self.dual_work.plus(out_dual)
        self.mult_lambda_AAT_lambda_inv(self.dual_work, out_dual)

        # compute the design term
        # out_design = in_design - VSU^T*out_dual
        self.mult_A_T_lambda(out_dual, out_design)
        out_design.times(-1.)
        out_design.plus(in_design)

        # compute the slack term
        # out_slack = in_slack + Sigma*Lambda*out_dual
        out_slack.equals(out_dual)
        out_slack.times(self.at_dual)
        out_slack.times(self.slack_term)
        out_slack.plus(in_slack)
        out_slack.restrict()

        # finally back out the actual dual term
        out_dual.times(self.at_dual)
