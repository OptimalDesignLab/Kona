
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.hessian.basic import BaseHessian

class TotalConstraintJacobian(BaseHessian):
    """
    Matrix object for the constraint block of the reduced KKT matrix.

    Uses the same 2nd order adjoint formulation as ReducedKKTMatrix, but only
    for the off-diagonal total contraint jacobian blocks,
    :math:`\\frac{dC}{dXd\\lambda}`.

    Parameters
    ----------
    use_design : boolean
        Flag to restrict design variables.
    use_target : boolean
        Flag to restrict target state variables.
    T : TotalConstraintJacobian
        Transposed matrix.
    approx : TotalConstraintJacobian
        Approximate/inexact matrix.
    """
    def __init__(self, vector_factories, optns={}):
        super(TotalConstraintJacobian, self).__init__(vector_factories, optns)

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
        self.use_design = True
        self.use_target = True
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
            self.adjoint = self.state_factory.generate()
            self.dual_work = self.dual_factory.generate()
            self._allocated = True

    def restrict_to_design(self):
        self.use_design = True
        self.use_target = False

    def restrict_to_target(self):
        self.use_design = False
        self.use_target = True

    def unrestrict(self):
        self.use_design = True
        self.use_target = True

    def product(self, in_vec, out_vec):
        if not self.use_design and not self.use_target:
            raise RuntimeError(
                'Both design and target components are set to false')

        # do some aliasing to make the code look pretty
        at_design = self.at_design
        at_state = self.at_state
        design_work = self.design_work
        state_work = self.state_work
        dual_work = self.dual_work
        adjoint = self.adjoint

        # compute out = A^T * (in)_(target subspace)
        if self._transposed:
            # convert input vector's target subspace into a dual vector
            dual_work.convert(in_vec)

            # compute (dC/dX)^T * dual_work and add to output vector
            dCdX(at_design, at_state).T.product(dual_work, design_work)
            out_vec.plus(design_work)

            # build RHS for adjoint system
            dCdU(at_design, at_state).T.product(dual_work, state_work)
            state_work.times(-1)

            # solve 2nd order adjoint
            adjoint.equals(0.0)
            if self._approx:
                # if this is approximate, use PDE preconditioner
                dRdU(at_design, at_state).T.precond(state_work, adjoint)
            else:
                # otherwise perform a full adjoint solution
                rel_tol = 1e-4
                dRdU(at_design, at_state).T.solve(state_work, adjoint, rel_tol)

            # apply lambda adjoint to design part of the Jacobian
            dRdX(at_design, at_state).T.product(adjoint, design_work)
            out_vec.plus(design_work)

            # restrict if necessary
            if self.use_design and not self.use_target:
                out_vec.restrict_to_design()
            if not self.use_design and self.use_target:
                out_vec.restrict_to_target()

        # compute (out)_(target subspace) = A * in
        else:
            # restrict if necessary
            design_work.equals(in_vec)
            if self.use_design and not self.use_target:
                design_work.restrict_to_design()
            if not self.use_design and self.use_target:
                design_work.restrict_to_target()

            # compute (dC/dX) * in
            dCdX(at_design, at_state).product(design_work, dual_work)
            out_vec.convert(dual_work)

            # build RHS for adjoint system
            dRdX(at_design, at_state).product(design_work, state_work)
            state_work.times(-1)

            # solve second order adjoint
            adjoint.equals(0.0)
            if self._approx:
                dRdU(at_design, at_state).precond(state_work, adjoint)
            else:
                rel_tol = 1e-4
                dRdU(at_design, at_state).solve(state_work, adjoint, rel_tol)

            # finish the dual part of the KKT matrix vector product
            dCdU(at_design, at_state).T.product(adjoint, dual_work)
            design_work.convert(dual_work)
            out_vec.plus(design_work)

        # reset the approx and transpose flags at the end
        self._approx = False
        self._transposed = False
