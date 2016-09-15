from kona.options import get_opt
from kona.linalg.vectors.common import DesignVector, StateVector, DualVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.hessian import LimitedMemorySR1
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import calc_epsilon, EPS

class QuasiNewtonKKTMatrix(BaseHessian):
    """
    An approximation of the KKT matrix using a limited memory SR1 for the
    Lagrangian Hessian, and a 2nd order adjoint formulation for the constraint
    jacobian.

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
        self.dual_factory = None
        for factory in self.vec_fac:
            if factory._vec_type is DesignVector:
                self.primal_factory = factory
            elif factory._vec_type is StateVector:
                self.state_factory = factory
            elif factory._vec_type is DualVector:
                self.dual_factory = factory

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
        self.dual_factory.request_num_vectors(3)

        # initialize abtract jacobians
        self.dRdX = dRdX()
        self.dRdU = dRdU()
        self.dCdX = dCdX()
        self.dCdU = dCdU()

    def set_krylov_solver(self, krylov_solver):
        if isinstance(krylov_solver, KrylovSolver):
            self.krylov = krylov_solver
        else:
            raise TypeError('Solver is not a valid KrylovSolver')

    def linearize(self, at_kkt, at_state):
        """
        An abstracted "linearization" method for the matrix.

        This method does not actually factor any real matrices. It also does
        not perform expensive linear or non-linear solves. It is used to update
        internal vector references and perform basic calculations using only
        cheap matrix-vector products.

        Parameters
        ----------
        at_design : DesignVector
            Design point at which the product is evaluated.
        at_state : StateVector
            State point at which the product is evaluated.
        at_dual : DualVector
            Lagrange multipliers at which the product is evaluated.
        at_adjoint : StateVector
            1st order adjoint variables at which the product is evaluated.
        barrer : float
            Log-barrier parameter used for slack equations.
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
            self.dual_work = self.dual_factory.generate()
            self.slack_work = self.dual_factory.generate()

            self._allocated = True

        # store the linearization point
        if isinstance(at_kkt.primal, CompositePrimalVector):
            self.at_design = at_kkt.primal.design
            self.at_slack = at_kkt.primal.slack
        else:
            self.at_design = at_kkt.primal
            self.at_slack = None
        self.design_norm = self.at_design.norm2
        self.at_state = at_state
        self.state_norm = self.at_state.norm2
        self.at_adjoint = at_adjoint
        self.at_dual = at_kkt.dual

        # linearize the constraint jacobian
        self.cnstr_jac.linearize(self.at_design, self.at_state)

    def add_correction(self, delta_x, grad_diff):
        self.quasi_newton(delta_x, grad_diff)

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
        if isinstance(in_vec.primal, CompositePrimalVector):
            if self.at_slack is None:
                raise TypeError('No slack variables defined!')
            in_design = in_vec.primal.design
            in_slack = in_vec.primal.slack
            out_design = out_vec.primal.design
            out_slack = out_vec.primal.slack
        else:
            in_design = in_vec.primal
            in_slack = None
            out_design = out_vec.primal
            out_slack = None
        in_dual = in_vec.dual
        out_dual = out_vec.dual

        # modify the in_vec for inequality constraints
        self.dual_work.equals_constraints(self.at_design, self.at_state)

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
        self.state_work[0].equals_objective_partial(
            self.pert_design, self.state_work[2])
        pert_state = self.state_work[2] # aliasing for readability
        self.dRdU.linearize(self.pert_design, pert_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[1])
        self.state_work[0].plus(self.state_work[1])
        self.dCdU.linearize(self.pert_design, pert_state)
        self.dCdU.T.product(self.at_dual, self.state_work[1])
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
        self.dRdX.linearize(self.pert_design, pert_state)
        self.dRdX.T.product(pert_adjoint, self.primal_work)
        out_design.plus(self.primal_work)
        self.dCdX.linearize(self.pert_design, pert_state)
        self.dCdX.T.product(self.at_dual, self.primal_work)
        out_design.plus(self.primal_work)

        # take difference with unperturbed conditions
        out_design.times(self.grad_scale)
        out_design.minus(self.reduced_grad)
        out_design.divide_by(epsilon_fd)

        # the dual part needs no FD
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.T.product(in_dual, self.primal_work)
        out_design.plus(self.primal_work)

        # evaluate dual part of product:
        # C = dC/dX*in_vec + dC/dU*w_adj
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.product(in_design, out_vec.dual)
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.product(self.w_adj, self.dual_work)
        out_dual.plus(self.dual_work)
        out_dual.times(self.feas_scale)

        # add globalization if necessary
        if self.lamb > EPS:
            out_design.equals_ax_p_by(
                1., out_design, self.lamb*self.scale, in_design)

        # add the slack term to the dual component
        if in_slack is not None:
            # compute slack term (-e^s)
            self.slack_work.exp(self.at_slack)
            self.slack_work.times(-1.)
            self.slack_work.restrict()
            # set slack output
            # out_slack = (-e^s * delta_lambda)
            out_slack.equals(in_dual)
            out_slack.times(self.slack_work)
            # add the slack contribution
            # out_slack += (-e^s * lambda * delta_slack)
            self.dual_work.equals(self.at_dual)
            self.dual_work.times(self.slack_work)
            self.dual_work.times(in_slack)
            out_slack.plus(self.dual_work)

            # add the slack contribution to dual component
            # out_dual += (-e^s * delta_slack)
            self.dual_work.equals(in_slack)
            self.dual_work.times(self.slack_work)
            out_dual.plus(self.dual_work)
