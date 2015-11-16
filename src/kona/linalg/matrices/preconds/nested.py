from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.solvers.krylov import FGMRES

class NestedKKTPreconditioner(ReducedKKTMatrix):
    """
    This object preconditions the KKT system by doing nested solves of the
    modified KKT system.

    The modified KKT system is given by:

    .. math::
        \\begin{bmatrix}
        W && A^T \\\\
        \\Lambda A && \\Sigma
        \\end{bmatrix}
        \\begin{bmatrix}
        v_p \\\\ v_d
        \\end{bmatrix} =
        \\begin{bmatrix}
        u_p \\\\ \\Lambda u_d - u_s
        \\end{bmatrix}

    where :math:`\\Lambda = diag(\\lambda)` and :math:`\\Sigma = diag(e^s)`.

    The matrix-vector product for the system above is assembled using the PDE
    preconditioner for the 2nd order adjoint solves.

    Once the system is solved, the preconditioned slack component is then
    backed out of the solution via
    :math:`v_s = -\\Sigma^{-1} \\Lambda^{-1} (u_s + \\Sigma v_d)`.
    """
    def __init__(self, vector_factories, optns={}):
        super(NestedKKTPreconditioner, self).__init__(vector_factories, optns)

        self.primal_factory.request_num_vectors(1)
        self.dual_factory.request_num_vectors(2)

        self.nested_allocated = False

        krylov_optns = {
            'out_file' : 'kona_nested_krylov.dat',
            'max_iter' : 200,
            'rel_tol'  : 1e-4,
        }
        self.krylov = FGMRES(
            self.primal_factory,
            optns=krylov_optns,
            dual_factory=self.dual_factory)

    def _linear_solve(self, rhs_vec, solution, rel_tol=1e-8):
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.precond(rhs_vec, solution)

    def _adjoint_solve(self, rhs_vec, solution, rel_tol=1e-8):
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.precond(rhs_vec, solution)

    def linearize(self, at_kkt, at_state, at_adjoint):
        super(NestedKKTPreconditioner, self).linearize(
            at_kkt, at_state, at_adjoint)

        if not self.nested_allocated:
            self.rhs_work = ReducedKKTVector(
                CompositePrimalVector(
                    self.primal_factory.generate(),
                    self.dual_factory.generate()),
                self.dual_factory.generate())
            self.nested_allocated = True

    def product(self, in_vec, out_vec):
        # strip the slack variables out of the in/out vectors
        short_in = ReducedKKTVector(in_vec._primal._design, in_vec._dual)
        short_out = ReducedKKTVector(out_vec._primal._design, out_vec._dual)

        # compute the slack-less KKT matrix-vector product
        super(NestedKKTPreconditioner, self).product(short_in, short_out)

        # set slack components of the output vecetor to zero
        out_vec._primal._slack.equals(0.0)

        # augment the dual component such that:
        # out_dual = diag(lambda)*A*in_design + diag(e^s)*in_dual
        out_vec._dual.times(self.at_dual)
        self.slack_work.exp(self.at_slack)
        self.slack_work.restrict()
        self.slack_work.times(in_vec._dual)
        out_vec._dual.plus(self.slack_work)

    def solve(self, rhs, solution, rel_tol=None):
        # make sure we have a krylov solver
        if self.krylov is None:
            raise AttributeError('krylov solver not set')

        # set tolerance
        if isinstance(rel_tol, float):
            self.krylov.rel_tol = rel_tol

        # define the no-op preconditioner
        eye = IdentityMatrix()
        precond = eye.product

        # modify the dual component of the RHS vector
        self.rhs_work.equals(rhs)
        self.rhs_work._dual.times(self.at_dual)
        rhs._primal._slack.restrict()
        self.rhs_work._dual.minus(rhs._primal._slack)
        # save slack component of RHS and then set it to zero
        self.rhs_work._primal._slack.equals(0.0)

        # solve the system
        self.krylov.solve(self.product, self.rhs_work, solution, precond)

        # back out the slack solutions
        self.dual_work.exp(self.at_slack)
        self.dual_work.restrict()
        solution._primal._slack.equals(solution._dual)
        solution._primal._slack.times(self.dual_work)
        solution._primal._slack.plus(rhs._primal._slack)
        self.dual_work.exp(self.at_slack)
        self.dual_work.pow(-1.)
        self.dual_work.restrict()
        solution._primal._slack.times(self.dual_work)
        self.dual_work.equals(self.at_dual)
        self.dual_work.pow(-1.)
        solution._primal._slack.times(self.dual_work)
        solution._primal._slack.times(-1.)
        solution._primal._slack.restrict()
