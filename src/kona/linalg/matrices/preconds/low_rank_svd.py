import numpy as np

from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.solvers.util import lanczos_bidiag, lanczos_tridiag
from kona.linalg.solvers.util import calc_epsilon

class LowRankSVD(BaseHessian):
    """
    This object preconditions the KKT system by solving a modified version of
    this system via a low-rank singular-value decomposition.

    The modified system is given as:

    .. math::
        \\begin{bmatrix}
        W && A^T \\Lambda \\\\
        \\Lambda A && \\Sigma \\Lambda
        \\end{bmatrix}
        \\begin{bmatrix}
        v_d \\\\ \\hat{v}_{\\lambda}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        u_d \\\\ \\Lambda (u_{\\lambda} - u_s)
        \\end{bmatrix}

    Here the :math:`A` matrix is the total constraint jacobian
    :math:`\\nabla_x c`, the :math:`W` matrix is the Hessian of the Lagrangian,
    :math:`\\Lambda = diag(\\lambda)` and :math:`\\Sigma = diag(e^s)`.

    :math:`(u_d, u_s, u_{\\lambda})^T` is the
    vector to be preconditioned, and :math:`(v_d, v_s, v_{\\lambda})^T` is the
    preconditioned vector. Finally,
    :math:`\\hat{v}_{\\lambda} = \\Lambda^{-1} v_{\\lambda}` is used as a
    substitution that helps us apply an better inverse approximation to the
    system.

    The hessian and constraint jacobian blocks are approximated using a
    low-rank SVD, formulated by applying the Lanczos bi-diagonalization
    algorithm to adjoint-based matrix-vector products. These 2nd order adjoints
    are approximately solved using the PDE preconditioner.

    This decomposition is performed once at the point where the KKT system is
    linearized, and then can be used repeatedly to produce cheap inverse
    approximations to the augmented system.
    """

    def __init__(self, vector_factories, optns={}):
        super(LowRankSVD, self).__init__(vector_factories, optns)

        # set basic internal options
        self.use_precond = True
        self.subspace_size = 10

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
        self.state_factory.request_num_vectors(6)
        self.dual_factory.request_num_vectors(3 + 2*self.subspace_size)

    def linearize(self, at_kkt, at_state, at_adjoint):

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
            self.pert_design = self.primal_factory.generate()
            self.design_work = self.primal_factory.generate()
            self.reduced_grad = self.primal_factory.generate()

            self.dual_work = self.dual_factory.generate()
            self.slack_term = self.dual_factory.generate()

            self.pert_state = self.state_factory.generate()
            self.state_work = self.state_factory.generate()
            self.adjoint_work = self.state_factory.generate()
            self.forward_adjoint = self.state_factory.generate()
            self.reverse_adjoint = self.state_factory.generate()
            self.adjoint_res = self.state_factory.generate()
            # flip the allocation flag
            self._allocated = True

        # do some aliasing
        if isinstance(at_kkt._primal, CompositePrimalVector):
            self.at_design = at_kkt._primal._design
            self.at_slack = at_kkt._primal._slack
        else:
            self.at_design = at_kkt._primal
            self.at_slack = None
        self.at_dual = at_kkt._dual
        self.at_state = at_state
        self.at_adjoint = at_adjoint
        self.slack_term.exp(self.at_slack)
        self.slack_term.restrict()

        # compute adjoint residual at the linearization
        self.dual_work.equals_constraints(self.at_design, self.at_state)
        self.adjoint_res.equals_objective_partial(self.at_design, self.at_state)
        dRdU(self.at_design, self.at_state).T.product(
            self.at_adjoint, self.state_work)
        self.adjoint_res.plus(self.state_work)
        dCdU(self.at_design, self.at_state).T.product(
            self.at_dual, self.state_work)
        self.adjoint_res.plus(self.state_work)

        # compute reduced gradient at the linearization
        self.reduced_grad.equals_objective_partial(
            self.at_design, self.at_state)
        dRdX(self.at_design, self.at_state).T.product(
            self.at_adjoint, self.design_work)
        self.reduced_grad.plus(self.design_work)
        dCdX(self.at_design, self.at_state).T.product(
            self.at_dual, self.design_work)
        self.reduced_grad.plus(self.design_work)

        # bi-diagonalize the constraint jacobian blocks
        B = lanczos_bidiag(self._multiply_A, self.Q, self.q_work,
                           self._multiply_A_T, self.P, self.p_work)

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

        # now recycle the subspace and tridiagonalize the Hessian block
        T = lanczos_tridiag(self._multiply_W, self.Q)
        self.E_left, gamma_tmp, E_right_T = np.linalg.svd(T, full_matrices=0)
        self.E_right = E_right_T.T
        self.gamma = np.diag(gamma_tmp)
        self.gamma_inv = np.linalg.inv(self.gamma)

    def _multiply_W(self, in_vec, out_vec):
        # calculate the FD perturbation for the design
        epsilon_fd = calc_epsilon(self.at_design.norm2, in_vec.norm2)

        # perturb the design variables
        self.pert_design.equals_ax_p_by(1.0, self.at_design, epsilon_fd, in_vec)

        # compute partial (d^2 L/dx^2)*in_vec and store in out_vec
        out_vec.equals_objective_partial(self.pert_design, self.at_state)
        dRdX(self.pert_design, self.at_state).T.product(
            self.at_adjoint, self.design_work)
        out_vec.plus(self.design_work)
        dCdX(self.pert_design, self.at_state).T.product(
            self.at_dual, self.design_work)
        out_vec.plus(self.design_work)
        out_vec.minus(self.reduced_grad)
        out_vec.divide_by(epsilon_fd)

        # build RHS for first adjoint system and solve for forward adjoint
        dRdX(self.at_design, self.at_state).product(in_vec, self.state_work)
        self.state_work.times(-1.)
        if self.use_precond:
            dRdU(self.at_design, self.at_state).precond(
                self.state_work, self.forward_adjoint)
        else:
            dRdU(self.at_design, self.at_state).solve(
                self.state_work, self.forward_adjoint, rel_tol=1e-8)

        # compute the FD perturbation for the states
        epsilon_fd = calc_epsilon(
            self.at_state.norm2, self.forward_adjoint.norm2)
        self.pert_state.equals_ax_p_by(
            1.0, self.at_state, epsilon_fd, self.forward_adjoint)

        # build RHS for second adjoint system

        # STEP 1: perturb design, evaluate adjoint residual, take difference
        self.adjoint_work.equals_objective_partial(
            self.pert_design, self.at_state)
        dRdU(self.pert_design, self.at_state).T.product(
            self.at_adjoint, self.state_work)
        self.adjoint_work.plus(self.state_work)
        dCdU(self.pert_design, self.at_state).T.product(
            self.at_dual, self.state_work)
        self.adjoint_work.plus(self.state_work)
        self.adjoint_work.minus(self.adjoint_res)
        self.adjoint_work.divide_by(epsilon_fd)

        # STEP 2: perturb state, evaluate adjoint residual, take difference
        self.reverse_adjoint.equals_objective_partial(
            self.at_design, self.pert_state)
        dRdU(self.at_design, self.pert_state).T.product(
            self.at_adjoint, self.state_work)
        self.reverse_adjoint.plus(self.state_work)
        dCdU(self.at_design, self.pert_state).T.product(
            self.at_dual, self.state_work)
        self.reverse_adjoint.plus(self.state_work)
        self.reverse_adjoint.minus(self.adjoint_res)
        self.reverse_adjoint.divide_by(epsilon_fd)

        # STEP 3: assemble the final RHS and solve the adjoint system
        self.adjoint_work.plus(self.reverse_adjoint)
        self.adjoint_work.times(-1.)
        if self.use_precond:
            dRdU(self.at_design, self.at_state).T.precond(
                self.adjoint_work, self.reverse_adjoint)
        else:
            dRdU(self.at_design, self.at_state).T.solve(
                self.adjoint_work, self.reverse_adjoint, rel_tol=1e-8)

        # now we can assemble the remaining pieces of the Hessian-vector product

        # apply reverse_adjoint to design part of the jacobian
        dRdX(self.at_design, self.at_state).T.product(
            self.reverse_adjoint, self.design_work)
        out_vec.plus(self.design_work)

        # apply the Lagrangian adjoint to the cross-derivative part of Hessian
        self.design_work.equals_objective_partial(
            self.at_design, self.pert_state)
        dRdX(self.at_design, self.pert_state).T.product(
            self.at_adjoint, self.q_work)
        self.design_work.plus(self.q_work)
        dCdX(self.at_design, self.pert_state).T.product(
            self.at_dual, self.q_work)
        self.design_work.plus(self.q_work)
        self.design_work.minus(self.reduced_grad)
        self.design_work.divide_by(epsilon_fd)
        out_vec.plus(self.design_work)

    def _multiply_A(self, in_vec, out_vec):
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
        out_vec.times(self.at_dual)

    def _multiply_A_T(self, in_vec, out_vec):
        # modify the input Vector
        self.dual_work.equals(in_vec)
        self.dual_work.times(self.at_dual)
        # assemble the RHS for the adjoint system
        dCdU(self.at_design, self.at_state).T.product(
            self.dual_work, self.state_work)
        self.state_work.times(-1.)
        # approximately solve the linear system
        if self.use_precond:
            dRdU(self.at_design, self.at_state).T.precond(
                self.state_work, self.adjoint_work)
        else:
            dRdU(self.at_design, self.at_state).T.solve(
                self.state_work, self.adjoint_work, rel_tol=1e-8)
        # assemble the final product
        dCdX(self.at_design, self.at_state).T.product(self.dual_work, out_vec)
        dRdX(self.at_design, self.at_state).T.product(
            self.adjoint_work, self.design_work)
        out_vec.plus(self.design_work)

    def approx_W_prod(self, in_vec, out_vec):
        QT_in = np.zeros(len(self.Q))
        for i in xrange(len(self.Q)):
            QT_in[i] = self.Q[i].inner(in_vec)
        E_right_T_QT_in = np.dot(self.E_right.T, QT_in)
        gamma_E_right_T_QT_in = np.dot(self.gamma, E_right_T_QT_in)
        TQT_in = np.dot(self.E_left, gamma_E_right_T_QT_in)
        out_vec.equals(0.0)
        for i in xrange(len(self.Q[:-1])):
            out_vec.equals_ax_p_by(1., out_vec, TQT_in[i], self.Q[i])

    def approx_A_prod(self, in_vec, out_vec):
        VT_in = np.zeros(len(self.V))
        for i in xrange(len(self.V)):
            VT_in[i] = self.V[i].inner(in_vec)
        SVT_in = np.dot(self.S, VT_in)
        out_vec.equals(0.0)
        for i in xrange(len(self.U)):
            out_vec.equals_ax_p_by(1., out_vec, SVT_in[i], self.U[i])

    def approx_AT_prod(self, in_vec, out_vec):
        UT_vec = np.zeros(len(self.U))
        for i in xrange(len(self.U)):
            UT_vec[i] = self.U[i].inner(in_vec)
        SUT_vec = np.dot(self.S, UT_vec)
        out_vec.equals(0.0)
        for i in xrange(len(self.V)):
            out_vec.equals_ax_p_by(1., out_vec, SUT_vec[i], self.V[i])
