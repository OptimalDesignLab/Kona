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
        super(TotalConstraintJacobian, self).__init__(vector_factories, None)

        # request vector allocation
        self.primal_factory.request_num_vectors(1)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(1)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(1)

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

    def linearize(self, at_design, at_state, scale=1.0):
        # store references to the evaluation point
        self.at_design = at_design
        self.at_state = at_state
        self.scale = 1.0

        # if this is the first linearization, produce some work vectors
        if not self._allocated:
            self.design_work = self.primal_factory.generate()
            self.state_work = self.state_factory.generate()
            self.adjoint_work = self.state_factory.generate()
            self.dual_work = None
            if self.eq_factory is not None and self.ineq_factory is not None:
                self.dual_work = CompositeDualVector(
                    self.eq_factory.generate(), self.ineq_factory.generate())
            elif self.eq_factory is not None:
                self.dual_work = self.eq_factory.generate()
            elif self.ineq_factory is not None:
                self.dual_work = self.ineq_factory.generate()
            else:
                raise RuntimeError(
                    "TotalConstraintJacobian >> " +
                    "Must have at least one type of dual vector factory!")
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
            dCdX(self.at_design, self.at_state).product(
                in_vec, out_vec)
            out_vec.times(self.scale)
            dCdU(self.at_design, self.at_state).product(
                self.adjoint_work, self.dual_work)
            self.dual_work.times(self.scale)
            out_vec.plus(self.dual_work)
        else:
            # assemble the RHS for the adjoint system
            dCdU(self.at_design, self.at_state).T.product(
                in_vec, self.state_work)
            self.state_work.times(self.scale)
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
            out_vec.times(self.scale)
            dRdX(self.at_design, self.at_state).T.product(
                self.adjoint_work, self.design_work)
            out_vec.plus(self.design_work)

        # reset the approx and transpose flags at the end
        self._approx = False
        self._transposed = False

# imports here to prevent circular errors
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU