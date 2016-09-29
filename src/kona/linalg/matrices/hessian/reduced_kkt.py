from kona.linalg.matrices.hessian.basic import BaseHessian

class ReducedKKTMatrix(BaseHessian):
    """
    Reduced approximation of the KKT matrix using a 2nd order adjoint
    formulation.

    The KKT system is defined as:

    .. math::
        \\begin{bmatrix}
        \\nabla_x^2 \\mathcal{L} && 0 && \\nabla_x c_{eq}^T && \\nabla_x c_{ineq}^T \\\\
        0 && \\Sigma && 0 && I \\\\
        \\nabla_x c_{eq} && 0 && 0 && 0 \\\\
        \\nabla_x c_{ineq} && I && 0 && 0
        \\end{bmatrix}
        \\begin{bmatrix}
        \\Delta x \\\\
        \\Delta s \\\\
        \\Delta \\lambda_{eq} \\\\
        \\Delta \\lambda_{ineq}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        -\\nabla_x \\mathcal{f} - \\lambda_{eq}^T \\nabla_x c_{eq} -\\lambda_{eq}^T \\nabla_x c_{eq} \\\\
        \\lambda^T \\mu * S^{-1}e \\\\
        - c_{eq} \\\\
        - c_{ineq} + s
        \\end{bmatrix}

    where :math:`\\mathcal{L}` is the Lagrangian defined as:

    .. math::
        \\mathcal{L}(x, u(x), \lambda) = F(x, u(x)) +
        \\lambda_{eq}^T c_{eq}(x, u(x)) +
        \\lambda_{ineq}^T \\left[c_{ineq}(x, u(x)) - s\\right] +
        \\frac{1}{2}\\mu\\sum_{i=1}^{n_{ineq}}ln(s_i)

    Inequality constrained are handled via the slack variables :math:`s` and
    the logarithmic barrier term enforcing non-negativity.

    .. note::

        More information on this 2nd order adjoint formulation can be found
        `in this paper <http://arc.aiaa.org/doi/abs/10.2514/6.2015-1945>`.

    Attributes
    ----------
    product_fac : float
    product_tol : float
    lamb : float
    scale : float
    grad_scale : float
    ceq_scale : float
    dynamic_tol : boolean
    krylov : KrylovSolver
        A krylov solver object used to solve the system defined by this matrix.
    dRdX, dRdU, dCdX, dCdU : KonaMatrix
        Various abstract jacobians used in calculating the mat-vec product.
    """
    def __init__(self, vector_factories, optns={}):
        super(ReducedKKTMatrix, self).__init__(vector_factories, optns)

        # read reduced options
        self.product_fac = get_opt(optns, 0.001, 'product_fac')
        self.product_tol = 1.0
        self.lamb = get_opt(optns, 0.0, 'lambda')
        self.scale = get_opt(optns, 1.0, 'scale')
        self.grad_scale = get_opt(optns, 1.0, 'grad_scale')
        self.feas_scale = get_opt(optns, 1.0, 'feas_scale')
        self.dynamic_tol = get_opt(optns, False, 'dynamic_tol')

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

        # request vector memory for future allocation
        self.primal_factory.request_num_vectors(3)
        self.state_factory.request_num_vectors(6)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(3)

        # initialize abtract jacobians
        self.dRdX = dRdX()
        self.dRdU = dRdU()
        self.dCdX = dCdX()
        self.dCdU = dCdU()

    def _linear_solve(self, rhs_vec, solution, rel_tol=1e-8):
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.solve(rhs_vec, solution, rel_tol=rel_tol)

    def _adjoint_solve(self, rhs_vec, solution, rel_tol=1e-8):
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.solve(rhs_vec, solution, rel_tol=rel_tol)

    def set_krylov_solver(self, krylov_solver):
        if isinstance(krylov_solver, KrylovSolver):
            self.krylov = krylov_solver
        else:
            raise TypeError('Solver is not a valid KrylovSolver')

    def linearize(self, at_kkt, at_state, at_adjoint,
                  obj_scale=1.0, cnstr_scale=1.0):
        """
        Linearize the KKT matrix at the given KKT, state, adjoint and barrier
        point. This method does not perform any factorizations or matrix
        operations.

        Parameters
        ----------
        at_kkt : ReducedKKTVector
            KKT vector at which the product is evaluated
        at_state : StateVector
            State point at which the product is evaluated.
        at_adjoint : StateVector
            1st order adjoint variables at which the product is evaluated.
        obj_scale : float, optional
            Factor by which the objective component of the product is scaled.
        cnstr_scale : float, optional
            Factor by which the constraint component of the product is scaled.
        """
        # if this is the first ever linearization...
        if not self._allocated:

            # generate state vectors
            self.adjoint_res = self.state_factory.generate()
            self.w_adj = self.state_factory.generate()
            self.lambda_adj = self.state_factory.generate()
            self.state_work = []
            for i in xrange(3):
                self.state_work.append(self.state_factory.generate())

            # generate primal vectors
            self.pert_design = self.primal_factory.generate()
            self.reduced_grad = self.primal_factory.generate()
            self.primal_work = self.primal_factory.generate()

            # generate dual vectors
            if isinstance(at_kkt.dual, CompositeDualVector):
                dual_eq = self.eq_factory.generate()
                dual_ineq = self.ineq_factory.generate()
                self.dual_work = CompositeDualVector(dual_eq, dual_ineq)
            else:
                self.dual_work = self.eq_factory.generate()
            self.slack_block = None
            if self.ineq_factory is not None:
                self.slack_block = self.ineq_factory.generate()

            self._allocated = True

        # store the linearization point
        self.at_dual = at_kkt.dual
        if isinstance(at_kkt.primal, CompositePrimalVector):
            self.at_design = at_kkt.primal.design
            self.at_slack = at_kkt.primal.slack
            if isinstance(self.at_dual, CompositeDualVector):
                self.at_dual_ineq = self.at_dual.ineq
            else:
                self.at_dual_ineq = self.at_dual
        else:
            self.at_design = at_kkt.primal
            self.at_slack = None
            self.at_dual_ineq = None
        self.design_norm = self.at_design.norm2
        self.at_state = at_state
        self.state_norm = self.at_state.norm2
        self.at_adjoint = at_adjoint
        self.at_dual = at_kkt.dual

        # store scales
        self.obj_scale = obj_scale
        self.cnstr_scale = cnstr_scale

        # pre compute the slack block
        if self.slack_block is not None:
            self.slack_block.equals(self.at_slack)
            self.slack_block.pow(-1.)
            self.slack_block.times(self.at_dual_ineq)

        # compute adjoint residual at the linearization
        self.dual_work.equals_constraints(self.at_design, self.at_state)
        self.adjoint_res.equals_objective_partial(self.at_design, self.at_state)
        self.adjoint_res.times(self.obj_scale)
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[0])
        self.adjoint_res.plus(self.state_work[0])
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.T.product(self.at_dual, self.state_work[0])
        self.state_work[0].times(self.cnstr_scale)
        self.adjoint_res.plus(self.state_work[0])

        # compute reduced gradient at the linearization
        self.reduced_grad.equals_objective_partial(self.at_design,
                                                   self.at_state)
        self.reduced_grad.times(self.obj_scale)
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.T.product(self.at_adjoint, self.primal_work)
        self.reduced_grad.plus(self.primal_work)
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.T.product(self.at_dual, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        self.reduced_grad.plus(self.primal_work)

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
        # type check given vectors
        if not isinstance(in_vec, ReducedKKTVector):
            raise TypeError('Multiplying vector is not a ReducedKKTVector')
        if not isinstance(out_vec, ReducedKKTVector):
            raise TypeError('Result vector is not a ReducedKKTVector')

        # clear output vector
        out_vec.equals(0.0)

        # do some aliasing to make the code cleanier
        in_dual = in_vec.dual
        out_dual = out_vec.dual
        if isinstance(in_vec.primal, CompositePrimalVector):
            if self.at_slack is None:
                raise TypeError('No slack variables defined!')
            in_design = in_vec.primal.design
            in_slack = in_vec.primal.slack
            out_design = out_vec.primal.design
            out_slack = out_vec.primal.slack
            if isinstance(in_dual, CompositeDualVector):
                in_dual_ineq = in_dual.ineq
                out_dual_ineq = out_dual.ineq
            else:
                in_dual_ineq = in_dual
                out_dual_ineq = out_dual
        else:
            in_design = in_vec.primal
            in_slack = None
            out_design = out_vec.primal
            out_slack = None
            in_dual_ineq = None
            out_dual_ineq = None

        # calculate appropriate FD perturbation for design
        epsilon_fd = calc_epsilon(self.design_norm, in_design.norm2)

        # assemble RHS for first adjoint system
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.product(in_design, self.state_work[0])
        self.state_work[0].times(-1.0)

        # perform the adjoint solution
        self.w_adj.equals(0.0)
        rel_tol = self.product_tol * \
            self.product_fac/max(self.state_work[0].norm2, EPS)
        # rel_tol = 1e-12
        self._linear_solve(self.state_work[0], self.w_adj, rel_tol=rel_tol)

        # find the adjoint perturbation by solving the linearized dual equation
        self.pert_design.equals_ax_p_by(
            1.0, self.at_design, epsilon_fd, in_design)
        self.state_work[2].equals_ax_p_by(
            1.0, self.at_state, epsilon_fd, self.w_adj)

        # first part of LHS: evaluate the adjoint equation residual at
        # perturbed design and state
        self.state_work[0].equals_objective_partial(self.pert_design,
                                                    self.state_work[2])
        self.state_work[0].times(self.obj_scale)
        pert_state = self.state_work[2] # aliasing for readability
        self.dRdU.linearize(self.pert_design, pert_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[1])
        self.state_work[0].plus(self.state_work[1])
        self.dCdU.linearize(self.pert_design, pert_state)
        self.dCdU.T.product(self.at_dual, self.state_work[1])
        self.state_work[1].times(self.cnstr_scale)
        self.state_work[0].plus(self.state_work[1])

        # at this point state_work[0] should contain the perturbed adjoint
        # residual, so take difference with unperturbed adjoint residual
        self.state_work[0].minus(self.adjoint_res)
        self.state_work[0].divide_by(epsilon_fd)

        # multiply by -1 to move to RHS
        self.state_work[0].times(-1.0)

        # second part of LHS: (dC/dU) * in_vec.dual
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.T.product(in_dual, self.state_work[1])
        self.state_work[1].times(self.cnstr_scale)

        # assemble final RHS
        self.state_work[0].minus(self.state_work[1])

        # perform the adjoint solution
        self.lambda_adj.equals(0.0)
        rel_tol = self.product_tol * \
            self.product_fac/max(self.state_work[0].norm2, EPS)
        # rel_tol = 1e-12
        self._adjoint_solve(
            self.state_work[0], self.lambda_adj, rel_tol=rel_tol)

        # evaluate first order optimality conditions at perturbed design, state
        # and adjoint:
        # g = df/dX + lag_mult*dC/dX + (adjoint + eps_fd*lambda_adj)*dR/dX
        self.state_work[1].equals_ax_p_by(
            1.0, self.at_adjoint, epsilon_fd, self.lambda_adj)
        pert_adjoint = self.state_work[1] # aliasing for readability
        out_design.equals_objective_partial(self.pert_design, pert_state)
        out_design.times(self.obj_scale)
        self.dRdX.linearize(self.pert_design, pert_state)
        self.dRdX.T.product(pert_adjoint, self.primal_work)
        out_design.plus(self.primal_work)
        self.dCdX.linearize(self.pert_design, pert_state)
        self.dCdX.T.product(self.at_dual, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        out_design.plus(self.primal_work)

        # take difference with unperturbed conditions
        out_design.times(self.grad_scale)
        out_design.minus(self.reduced_grad)
        out_design.divide_by(epsilon_fd)

        # the dual part needs no FD
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.T.product(in_dual, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        out_design.plus(self.primal_work)

        # evaluate dual part of product:
        # C = dC/dX*in_vec + dC/dU*w_adj
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.product(in_design, out_vec.dual)
        out_vec.dual.times(self.cnstr_scale)
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.product(self.w_adj, self.dual_work)
        self.dual_work.times(self.cnstr_scale)
        out_dual.plus(self.dual_work)
        out_dual.times(self.feas_scale)

        # add globalization if necessary
        if self.lamb > EPS:
            out_design.equals_ax_p_by(
                1., out_design, self.lamb*self.scale, in_design)

        # add the slack term to the dual component
        if in_slack is not None:
            # set slack output
            # out_slack = Sigma * in_slack + in_dual_ineq
            out_slack.equals(in_slack)
            out_slack.times(self.slack_block)
            out_slack.plus(in_dual_ineq)
            # add the slack contribution to dual component
            # out_dual_ineq += in_slack
            out_dual_ineq.plus(in_slack)

# imports here to prevent circular errors
from numbers import Number
from kona.options import get_opt
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import calc_epsilon, EPS