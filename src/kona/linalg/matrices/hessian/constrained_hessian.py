
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.solvers.util import calc_epsilon

class ConstrainedHessian(BaseHessian):
    """
    Matrix object for the Hessian block of the reduced KKT matrix.

    Uses the same 2nd order adjoint formulation as ReducedKKTMatrix, but only
    for the diagonal Hessian block,
    :math:`\mathsf{W} = \\nabla^2_x \\mathcal{L}`.
    """
    def __init__(self, vector_factories):
        super(ConstrainedHessian, self).__init__(vector_factories, {})

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

        # request vector allocation
        self.primal_factory.request_num_vectors(3)
        self.state_factory.request_num_vectors(6)
        self.dual_factory.request_num_vectors(1)

        # set misc settings
        self._approx = False
        self._transposed = False
        self._allocated = False

    @property
    def approx(self):
        self._approx = True
        return self

    def linearize(self, at_primal, at_dual, at_state, at_adjoint):
        # store references to the evaluation point
        self.at_design = at_primal
        self.at_dual = at_dual
        self.at_state = at_state
        self.at_adjoint = at_adjoint

        # if this is the first linearization, produce some work vectors
        if not self._allocated:
            self.design_work = self.primal_factory.generate()
            self.reduced_grad = self.primal_factory.generate()
            self.pert_design = self.primal_factory.generate()
            self.state_work = self.state_factory.generate()
            self.adjoint_work = self.state_factory.generate()
            self.forward_adjoint = self.state_factory.generate()
            self.reverse_adjoint = self.state_factory.generate()
            self.adjoint_res = self.state_factory.generate()
            self.pert_state = self.state_factory.generate()
            self.dual_work = self.dual_factory.generate()
            self._allocated = True

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

    def product(self, in_vec, out_vec):
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
        if self._approx:
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
        if self._approx:
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
            self.at_adjoint, self.pert_design)
        self.design_work.plus(self.pert_design)
        dCdX(self.at_design, self.pert_state).T.product(
            self.at_dual, self.pert_design)
        self.design_work.plus(self.pert_design)
        self.design_work.minus(self.reduced_grad)
        self.design_work.divide_by(epsilon_fd)
        out_vec.plus(self.design_work)

        # reset the approx and transpose flags at the end
        self._approx = False
