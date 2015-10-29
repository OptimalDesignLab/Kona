from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix

class NestedKKTPreconditioner(ReducedKKTMatrix):
    """
    This object preconditions the KKT system by doing approximate solutions
    of the 2nd order adjoints using the PDE preconditioner.

    The approximate product using the approximate adjoints are then used in a
    nested Krylov solver to produce an inverse estimate.
    """
    # def _linear_solve(self, rhs_vec, solution, rel_tol=1e-8):
    #     self.dRdU.linearize(self.at_design, self.at_state)
    #     self.dRdU.precond(rhs_vec, solution)
    #
    # def _adjoint_solve(self, rhs_vec, solution, rel_tol=1e-8):
    #     self.dRdU.linearize(self.at_design, self.at_state)
    #     self.dRdU.T.precond(rhs_vec, solution)

    def solve(self, rhs, solution, rel_tol=None):
        # make sure we have a krylov solver
        if self.krylov is None:
            raise AttributeError('krylov solver not set')

        # set tolerance
        if isinstance(rel_tol, float):
            self.krylov.rel_tol = rel_tol

        # define the preconditioner
        eye = IdentityMatrix()
        precond = eye.product

        # trigger the solution
        self.krylov.solve(self.product, rhs, solution, precond)
