
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.hessian.basic import BaseHessian

class TotalConstraintJacobian(BaseHessian):
    """
    Matrix object for the constraint block of the reduced KKT matrix.

    Uses the same 2nd order adjoint formulation as ReducedKKTMatrix, but only
    for the off-diagonal total contraint jacobian blocks,
    :math:`\mathsf{A} = \\nabla_x C`.

    Parameters
    ----------
    T : TotalConstraintJacobian
        Transposed matrix.
    approx : TotalConstraintJacobian
        Approximate/inexact matrix.
    """
    def __init__(self, vector_factories):
        super(TotalConstraintJacobian, self).__init__(vector_factories, {})

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

        # request vector allocation
        self.primal_factory.request_num_vectors(1)
        self.state_factory.request_num_vectors(2)
        self.dual_factory.request_num_vectors(1)

        # set misc settings
        self._approx = False
        self._transposed = False
        self._allocated = False

    @property
    def approx(self):
        self._approx = True
        return self

    @property
    def T(self):
        self._transposed = True
        return self

    def linearize(self, at_design, at_state):
        # store references to the evaluation point
        self.at_design = at_design
        self.at_state = at_state

        # if this is the first linearization, produce some work vectors
        if not self._allocated:
            self.design_work = self.primal_factory.generate()
            self.state_work = self.state_factory.generate()
            self.adjoint_work = self.state_factory.generate()
            self.dual_work = self.dual_factory.generate()
            self._allocated = True

    def product(self, in_vec, out_vec):
        if not self._transposed:
            # assemble the RHS for the linear system
            dRdX(self.at_design, self.at_state).product(
                in_vec, self.state_work)
            self.state_work.times(-1.)
            # approximately solve the linear system
            if self._approx:
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

        else:
            # assemble the RHS for the adjoint system
            dCdU(self.at_design, self.at_state).T.product(
                in_vec, self.state_work)
            self.state_work.times(-1.)
            # approximately solve the linear system
            if self._approx:
                dRdU(self.at_design, self.at_state).T.precond(
                    self.state_work, self.adjoint_work)
            else:
                dRdU(self.at_design, self.at_state).T.solve(
                    self.state_work, self.adjoint_work, rel_tol=1e-8)
            # assemble the final product
            dCdX(self.at_design, self.at_state).T.product(
                in_vec, out_vec)
            dRdX(self.at_design, self.at_state).T.product(
                self.adjoint_work, self.design_work)
            out_vec.plus(self.design_work)

        # reset the approx and transpose flags at the end
        self._approx = False
        self._transposed = False
