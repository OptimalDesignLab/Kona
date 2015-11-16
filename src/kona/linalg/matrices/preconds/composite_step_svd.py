import numpy as np

from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.solvers.util import lanczos

class CompositeStepSVD(BaseHessian):
    """
    This object preconditions the augmented system with a composite-step
    solution calculated using a matrix-free low-rank singular-value
    decomposition of the constraint jacobian.

    The augmented system is given as:

    .. math::
        \\begin{bmatrix}
        I && 0 && A^T \\\\
        0 && I && 0 \\\\
        A && 0 && 0
        \\end{bmatrix}

    Here the :math:`A` matrix is the total constraint jacobian
    :math:`\\nabla_x c`.

    The constraint jacobian is approximated using a low-rank SVD, formulated
    by applying the Lanczos bi-diagonalization algorithm to adjoint-based
    matrix-vector products.

    This decomposition is performed once at the point where the KKT system is
    linearized, and then can be used repeatedly to produce cheap inverse
    approximations to the augmented system.
    """
    def __init__(self, vector_factories, optns={}):
        super(CompositeStepSVD, self).__init__(vector_factories, optns)

        # set basic internal options
        self.use_precond = True
        self.subspace_size = 20
        self.P = []
        self.V = []

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

        # reset the linearization flag
        self._allocated = False

        # request vector memory for future allocation
        self.primal_factory.request_num_vectors(5 + 2*self.subspace_size)
        self.state_factory.request_num_vectors(2)
        self.dual_factory.request_num_vectors(2 + 2*self.subspace_size)

    def _fwd_mat_vec(self, in_vec, out_vec):
        # assemble the RHS for the linear system
        dRdX(self.at_design, self.at_state).product(
            in_vec, self.state_work)
        self.state_work.times(-1.)
        # approximately solve the linear system
        if self.use_precond:
            dRdU(self.at_design, self.at_state).precond(
                self.state_work, self.adjoint_work)
        else:
            dRdU(self.at_design, self.at_state).solve(
                self.state_work, self.adjoint_work, rel_tol=1e-8)
        # assemble the product
        dCdX(self.at_design, self.at_state).product(in_vec, out_vec)
        dCdU(self.at_design, self.at_state).product(
            self.adjoint_work, self.dual_work)
        out_vec.plus(self.dual_work)

    def _rev_mat_vec(self, in_vec, out_vec):
        # assemble the RHS for the adjoint system
        dCdU(self.at_design, self.at_state).T.product(
            in_vec, self.state_work)
        self.state_work.times(-1.)
        # approximately solve the linear system
        if self.use_precond:
            dRdU(self.at_design, self.at_state).T.precond(
                self.state_work, self.adjoint_work)
        else:
            dRdU(self.at_design, self.at_state).T.solve(
                self.state_work, self.adjoint_work, rel_tol=1e-8)
        # assemble the final product
        dCdX(self.at_design, self.at_state).T.product(in_vec, out_vec)
        dRdX(self.at_design, self.at_state).T.product(
            self.adjoint_work, self.design_work)
        out_vec.plus(self.design_work)

    def linearize(self, at_design, at_state):

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
            self.normal_design = self.primal_factory.generate()
            self.tangent_design = self.primal_factory.generate()
            self.design_work = self.primal_factory.generate()
            self.dual_work = self.dual_factory.generate()
            self.state_work = self.state_factory.generate()
            self.adjoint_work = self.state_factory.generate()
            # flip the allocation flag
            self._allocated = True

        # do some aliasing
        self.at_design = at_design
        self.at_state = at_state

        # run the Lanczos bi-diagonalization
        B = lanczos(self._fwd_mat_vec, self.Q, self.q_work,
                    self._rev_mat_vec, self.P, self.p_work)

        # decompose the bi-diagonal matrix
        u_tmp, s_tmp, v_tmp = np.linalg.svd(B)

        # diagonalize the singular values and calculate inverse
        self.S_inv = np.linalg.inv(np.diag(s_tmp))

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

    def product(self, in_vec, out_vec):
        # we don't modify slacks, so just pass them on
        out_vec._primal._slack.equals(in_vec._primal._slack)

        # do some aliasing
        in_design = in_vec._primal._design
        in_dual = in_vec._dual
        out_design = out_vec._primal._design
        out_dual = out_vec._dual

        # compute the normal primal step
        # STEP 1: U^T * in_vec_primal
        UT_in = np.zeros(len(self.U))
        for i in xrange(len(self.U)):
            UT_in[i] = self.U[i].inner(in_dual)
        # STEP 2: S^-1 * UT_in
        S_inv_UT_in = np.dot(self.S_inv, UT_in)
        # STEP 3: V * S_inv_UT_in
        self.normal_design.equals(0.0)
        for i in xrange(len(self.V)):
            self.design_work.equals(self.V[i])
            self.design_work.times(S_inv_UT_in[i])
            self.normal_design.plus(self.design_work)

        # compute the tangent primal step
        # STEP 1: V^T * in_design
        VT_in = np.zeros(len(self.V))
        for i in xrange(len(self.V)):
            VT_in[i] = self.V[i].inner(in_design)
        # STEP 2: V * VT_in
        self.tangent_design.equals(0.0)
        for i in xrange(len(self.V)):
            self.design_work.equals(self.V[i])
            self.design_work.times(VT_in[i])
            self.tangent_design.plus(self.design_work)
        # STEP 3: in_design - VVT_in
        self.tangent_design.times(-1.)
        self.tangent_design.plus(in_design)

        # compute the complete primal step
        out_design.equals_ax_p_by(
            1.0, self.normal_design, 1.0, self.tangent_design)

        # now compute the dual step
        # STEP 1: compute in_design - out_design
        self.design_work.equals_ax_p_by(1.0, in_design, 1.0, out_design)
        # STEP 2: V^T * (in_design - out_design)
        VT_vec = np.zeros(len(self.V))
        for i in xrange(len(self.V)):
            VT_vec[i] = self.V[i].inner(self.design_work)
        # STEP 3: S^-1 * VT_vec
        S_inv_VT_vec = np.dot(self.S_inv, VT_vec)
        # STEP 4: U * S_inv_VT_vec
        out_dual.equals(0.0)
        for i in xrange(len(self.U)):
            self.dual_work.equals(self.U[i])
            self.dual_work.times(S_inv_VT_vec[i])
            out_dual.plus(self.dual_work)
