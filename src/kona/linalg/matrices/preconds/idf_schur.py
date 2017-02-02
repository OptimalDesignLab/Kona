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
        krylov_opts = {
            'subspace_size' : 5,
            'rel_tol' : 1e-2,
            'check_res' :  False,
            'check_LS_grad' : False,
            'krylov_file' : KonaFile(
                'kona_schur.dat', self.primal_factory._memory.rank)}
        self.krylov = FGMRES(self.primal_factory, optns=krylov_opts)

        # initialize an identity preconditioner
        self.eye = IdentityMatrix()
        self.precond = self.eye.product

        # initialize the total constraint jacobian block
        self.cnstr_jac = TotalConstraintJacobian(vector_factories)

        # set misc settings
        self.diag = 0.0
        self._allocated = False

    def prod_target(self, in_vec, out_vec):
        self.design_prod.equals(in_vec)
        self.design_prod.restrict_to_target()
        self.cnstr_jac.approx.product(self.design_prod, self.dual_prod)
        out_vec.equals(0.0)
        self.dual_prod.convert_to_design(out_vec)

    def prod_target_t(self, in_vec, out_vec):
        self.dual_prod.equals(0.0)
        in_vec.convert_to_dual(self.dual_prod)
        self.cnstr_jac.T.approx.product(self.dual_prod, out_vec)
        out_vec.restrict_to_target()

    def linearize(self, at_primal, at_state, scale=1.0):
        # store references to the evaluation point
        if isinstance(at_primal, CompositePrimalVector):
            self.at_design = at_primal.design
        else:
            self.at_design = at_primal
        self.at_state = at_state

        # save the scaling on constraint terms
        self.scale = scale

        # linearize the constraint jacobian
        self.cnstr_jac.linearize(self.at_design, self.at_state, scale=self.scale)

        # if this is the first linearization, allocate some useful vectors
        if not self._allocated:
            # design vectors
            self.design_prod = self.primal_factory.generate()
            self.design_work = []
            for i in xrange(2):
                self.design_work.append(self.primal_factory.generate())
            # dual vectors
            self.dual_prod = None
            if self.eq_factory is not None and self.ineq_factory is not None:
                self.dual_prod = CompositeDualVector(
                    self.eq_factory.generate(), self.ineq_factory.generate())
            else:
                self.dual_prod = self.eq_factory.generate()

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

        out_design.equals(0.0)
        out_dual.equals(0.0)

        # Step 1: Solve A_targ^T * v_dual = u_targ
        design_work[1].equals(in_design)
        design_work[1].restrict_to_target()
        design_work[0].equals(0.0)
        self.prod_target_t(design_work[1], design_work[0])
        self.krylov.solve(
            self.prod_target_t, design_work[1], design_work[0], self.precond)
        design_work[0].convert_to_dual(out_dual)
        
        # Step 2: Compute v_x = u_x - A_x^T * v_dual
        design_work[0].equals(0.0)
        self.cnstr_jac.T.approx.product(out_dual, design_work[0])
        out_design.equals_ax_p_by(1., in_design, -1., design_work[0])
        out_design.restrict_to_design()

        # Step 3: Solve A_targ * v_targ = u_dual - A_x * v_x
        self.dual_prod.equals(0.0)
        self.cnstr_jac.approx.product(out_design, self.dual_prod)
        self.dual_prod.equals_ax_p_by(1., in_dual, -1., self.dual_prod)
        self.dual_prod.convert_to_design(design_work[1])
        design_work[1].restrict_to_target()
        design_work[0].equals(0.0)
        self.krylov.solve(
            self.prod_target, design_work[1], design_work[0], self.precond)
        design_work[0].restrict_to_target()
        out_design.plus(design_work[0])
    

# imports here to prevent circular errors
import numpy as np
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.memory import KonaFile