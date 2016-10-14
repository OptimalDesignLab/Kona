from kona.linalg.matrices.hessian.basic import BaseHessian

class ReducedSchurPreconditioner(BaseHessian):
    """
    An IDF-Schur preconditioner designed to precondition the KKT system for
    multidisciplinary design optimization problems formulated using the IDF
    architecture.

    The preconditioner solves a system defined by the matrix:

    .. math::
        \\begin{bmatrix} I && A^T \\\\ A && 0 \\end{bmatrix}

    This solution is used as the preconditioner to the complete KKT system.

    Unlike the complete KKT system, this solution can be performed using FGMRES.

    Attributes
    ----------
    krylov : KrylovSolver
    cnstr_jac : TotalConstraintJacobian

    """
    def __init__(self, vector_factories, optns=None):
        super(ReducedSchurPreconditioner, self).__init__(
            vector_factories, optns)

        self.primal_factory.request_num_vectors(3)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(1)
        else:
            raise RuntimeError(
                    "ReducedSchurPreconditioner >> " +
                    "Problem must have equality constraints!")
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(1)

        # initialize the internal FGMRES solver
        self.krylov = FGMRES(self.primal_factory)
        self.krylov.out_file = KonaFile('kona_schur.dat', 0)
        self.max_iter = 15

        # initialize an identity preconditioner
        self.eye = IdentityMatrix()
        self.precond = self.eye.product

        # initialize the total constraint jacobian block
        self.cnstr_jac = TotalConstraintJacobian(vector_factories)

        # set misc settings
        self.diag = 0.0
        self._allocated = False

    def prod_design(self, in_vec, out_vec):
        # in_vec = DesignVector
        # out_vec = DesignVector
        self.design_prod.equals(in_vec)
        self.design_prod.restrict_to_design()
        self.cnstr_jac.approx.product(self.design_prod, self.dual_prod)
        self.dual_prod.convert_to_design(out_vec)

    def prod_target(self, in_vec, out_vec):
        # in_vec = DesignVector
        # out_vec = DesignVector
        self.design_prod.equals(in_vec)
        self.design_prod.restrict_to_target()
        self.cnstr_jac.approx.product(self.design_prod, self.dual_prod)
        self.dual_prod.convert_to_design(out_vec)

    def prod_design_t(self, in_vec, out_vec):
        # in_vec = DualVectorEQ or CompositeDualVector
        # out_vec = DualVectorEQ or CompositeDualVector
        in_vec.convert_to_dual(self.dual_prod)
        self.cnstr_jac.T.approx.product(self.dual_prod, out_vec)
        out_vec.restrict_to_design()

    def prod_target_t(self, in_vec, out_vec):
        # in_vec = DualVectorEQ or CompositeDualVector
        # out_vec = DualVectorEQ or CompositeDualVector
        in_vec.convert_to_dual(self.dual_prod)
        self.cnstr_jac.T.approx.product(self.dual_prod, out_vec)
        out_vec.restrict_to_target()

    def linearize(self, at_KKT, at_state, scale=1.0):
        # store references to the evaluation point
        try:
            self.at_design = at_KKT.primal.design
            self.at_slack = at_KKT.primal.slack
        except Exception:
            self.at_design = at_KKT.primal
            self.at_slack = None
        self.at_state = at_state
        self.at_dual = at_KKT.dual
        self.at_KKT = at_KKT

        # save the scaling on constraint terms
        self.scale = scale

        # linearize the constraint jacobian
        self.cnstr_jac.linearize(self.at_design, self.at_state, scale=self.scale)

        # if this is the first linearization, allocate some useful vectors
        if not self._allocated:
            self.design_prod = self.primal_factory.generate()
            self.dual_prod = None
            if self.eq_factory is not None and self.ineq_factory is not None:
                self.dual_prod = CompositeDualVector(
                    self.eq_factory.generate(), self.ineq_factory.generate())
            else:
                self.dual_prod = self.eq_factory.generate()
            self.design_work = []
            for i in xrange(2):
                self.design_work.append(self.primal_factory.generate())

    def product(self, in_vec, out_vec):
        # do some aliasing
        try:
            in_design = in_vec.primal.design
            out_design = out_vec.primal.design
            out_vec.primal.slack.equals(in_vec.primal.slack)
        except Exception:
            in_design = in_vec.primal
            out_design = out_vec.primal
        in_dual = in_vec.dual
        out_dual = out_vec.dual
        design_work = self.design_work

        # set solver settings
        rel_tol = 0.01
        self.krylov.rel_tol = rel_tol
        self.krylov.check_res = False

        out_design.equals(0.0)
        out_dual.equals(0.0)

        # Step 1: Solve (dC/dy)^T in_dual = (u_design)_(target subspace)
        design_work[1].equals(in_design)
        design_work[1].restrict_to_target()
        design_work[0].equals(0.0)
        self.krylov.solve(
            self.prod_target_t, design_work[1], design_work[0], self.precond)
        design_work[0].convert_to_dual(out_dual)

        # Step 2: Compute (out_design)_(design subspace) =
        # (in_design)_(design subspace) - (dC/dx)^T * out_dual
        self.prod_design_t(design_work[0], out_design)
        fac = 1.0 # /(1.0 + self.diag)
        out_design.equals_ax_p_by(-fac, out_design, fac, in_design)
        out_design.restrict_to_design()

        # Step 3: Solve (dC/dy) (out_design)_(target subspace) =
        # in_dual - (dC/dx) (out_design)_(design subspace)
        self.prod_design(out_design, design_work[0])
        in_dual.convert_to_design(design_work[1])
        design_work[0].equals_ax_p_by(-1., design_work[0], 1., design_work[1])
        design_work[1].equals(0.0)
        self.krylov.solve(
            self.prod_target, design_work[0], design_work[1], self.precond)
        out_design.plus(design_work[1])

# imports here to prevent circular errors
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.memory import KonaFile