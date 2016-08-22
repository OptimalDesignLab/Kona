from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.memory import KonaFile

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
    def __init__(self, vector_factories, optns={}):
        super(ReducedSchurPreconditioner, self).__init__(
            vector_factories, optns)

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

        self.primal_factory.request_num_vectors(3)
        self.dual_factory.request_num_vectors(1)

        # initialize the internal FGMRES solver
        self.krylov = FGMRES(self.primal_factory, {})
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
        self.design_prod.equals(in_vec)
        self.design_prod.restrict_to_design()
        self.cnstr_jac.approx.product(self.design_prod, self.dual_prod)
        out_vec.equals(0.0)
        out_vec.convert(self.dual_prod)

    def prod_target(self, in_vec, out_vec):
        self.design_prod.equals(in_vec)
        self.design_prod.restrict_to_target()
        self.cnstr_jac.approx.product(self.design_prod, self.dual_prod)
        out_vec.equals(0.0)
        out_vec.convert(self.dual_prod)

    def prod_design_t(self, in_vec, out_vec):
        self.dual_prod.equals(0.0)
        self.dual_prod.convert(in_vec)
        self.cnstr_jac.T.approx.product(self.dual_prod, out_vec)
        out_vec.restrict_to_design()

    def prod_target_t(self, in_vec, out_vec):
        self.dual_prod.equals(0.0)
        self.dual_prod.convert(in_vec)
        self.cnstr_jac.T.approx.product(self.dual_prod, out_vec)
        out_vec.restrict_to_target()

    def linearize(self, at_KKT, at_state):
        # store references to the evaluation point
        try:
            self.at_design = at_KKT._primal._design
            self.at_slack = at_KKT._primal._slack
        except Exception:
            self.at_design = at_KKT._primal
            self.at_slack = None
        self.at_state = at_state
        self.at_dual = at_KKT._dual
        self.at_KKT = at_KKT

        # linearize the constraint jacobian
        self.cnstr_jac.linearize(self.at_design, self.at_state)

        # if this is the first linearization, allocate some useful vectors
        if not self._allocated:
            self.design_prod = self.primal_factory.generate()
            self.dual_prod = self.dual_factory.generate()
            self.design_work = []
            for i in xrange(2):
                self.design_work.append(self.primal_factory.generate())

    def product(self, in_vec, out_vec):
        # do some aliasing
        try:
            in_design = in_vec._primal._design
            out_design = out_vec._primal._design
            out_vec._primal._slack.equals(in_vec._primal._slack)
        except Exception:
            in_design = in_vec._primal
            out_design = out_vec._primal
        in_dual = in_vec._dual
        out_dual = out_vec._dual
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
        out_dual.convert(design_work[0])

        # Step 2: Compute (out_design)_(design subspace) =
        # (in_design)_(design subspace) - (dC/dx)^T * out_dual
        self.prod_design_t(design_work[0], out_design)
        fac = 1.0 # /(1.0 + self.diag)
        out_design.equals_ax_p_by(-fac, out_design, fac, in_design)
        out_design.restrict_to_design()

        # Step 3: Solve (dC/dy) (out_design)_(target subspace) =
        # in_dual - (dC/dx) (out_design)_(design subspace)
        self.prod_design(out_design, design_work[0])
        design_work[1].convert(in_dual)
        design_work[0].equals_ax_p_by(-1., design_work[0], 1., design_work[1])
        design_work[1].equals(0.0)
        self.krylov.solve(
            self.prod_target, design_work[0], design_work[1], self.precond)
        out_design.plus(design_work[1])
