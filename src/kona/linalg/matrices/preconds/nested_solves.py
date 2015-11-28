from kona.options import get_opt
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.matrices.preconds import LowRankSVD
from kona.linalg.solvers.krylov import GCROT, FGMRES

class NestedNormalStepPreconditioner(object):

    def __init__(self, primal_factory, dual_factory, A, optns={}):

        # get vector factories
        self.primal_factory = primal_factory
        self.dual_factory = dual_factory

        # request a dual vector
        self.dual_factory.request_num_vectors(2)

        # set the allocation flag
        self._allocated = False

        # get the constraint jacobian object
        self.A = A

        # initialize the low-rank SVD approximation
        svd_optns = {
            'lanczos_size' : get_opt(optns, 10, 'lanczos_size'),
        }
        self.svd = LowRankSVD(
            self._fwd_mat_vec, self.primal_factory,
            self._rev_mat_vec, self.dual_factory, svd_optns)

        # initialize the internal krylov solver
        krylov_optns = {
            'out_file' : 'kona_nested_krylov.dat',
            'subspace_size' : 10,
            'max_recycle' : 10,
            'max_outer' : 10,
            'max_matvec' : 100,
            'check_res' : True,
            'rel_tol' : 1e-3,
            'abs_tol' : 1e-5,
        }
        self.krylov = GCROT(
            self.primal_factory,
            optns=krylov_optns,
            dual_factory=self.dual_factory)

    def _fwd_mat_vec(self, in_vec, out_vec):
        self.A.product(in_vec, out_vec)
        out_vec.times(self.at_dual)

    def _rev_mat_vec(self, in_vec, out_vec):
        self.dual_work.equals(in_vec)
        self.dual_work.times(self.at_dual)
        self.A.T.product(self.dual_work, out_vec)

    def linearize(self, at_dual, at_slack):
        # if this is the first linearization, create a work vector
        if not self._allocated:
            self.dual_work = self.dual_factory.generate()
            self.slack_term = self.dual_factory.generate()
            self._allocated = True

        # get the dual linearization
        self.at_dual = at_dual
        self.at_slack = at_slack
        self.slack_term.exp(self.at_slack)
        self.slack_term.restrict()

        # trigger the SVD decomposition
        self.svd.linearize()

        # reset GCROT cycled subspace
        self.krylov.clear_subspace()

    def product(self, in_vec, out_vec):
        # calculate the design component
        self.svd.approx_rev_prod(in_vec._dual, out_vec._primal._design)
        out_vec._primal._design.plus(in_vec._primal._design)

        # calculate the slack component
        out_vec._primal._slack.equals(in_vec._dual)
        out_vec._primal._slack.times(self.at_dual)
        out_vec._primal._slack.times(self.slack_term)
        out_vec._primal._slack.times(-1.)
        out_vec._primal._slack.plus(in_vec._primal._slack)

        # calculate the dual component
        self.svd.approx_fwd_prod(in_vec._primal._design, out_vec._dual)
        self.dual_work.equals(in_vec._primal._slack)
        self.dual_work.times(self.at_dual)
        self.dual_work.times(self.slack_term)
        self.dual_work.times(-1.)
        out_vec._dual.plus(self.dual_work)

    def solve(self, rhs_vec, solution, rel_tol=None):
        # set the tolerance for the krylov solution
        if rel_tol is not None:
            tmp_rel_tol = self.krylov.rel_tol
            self.krylov.rel_tol = rel_tol

        # get a no-op identity preconditioner
        eye = IdentityMatrix()
        precond = eye.product

        # trigger the solution
        solution.equals(0.0)
        self.krylov.solve(self.product, rhs_vec, solution, precond)

        # back out the corect dual solution
        solution._dual.times(self.at_dual)

        # reset the tolerance
        if rel_tol is not None:
            self.krylov.rel_tol = tmp_rel_tol

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

        self.primal_factory.request_num_vectors(2)
        self.dual_factory.request_num_vectors(4)

        self.nested_allocated = False
        self.use_gcrot = True

        if self.use_gcrot:
            krylov_optns = {
                'out_file' : 'kona_nested_krylov.dat',
                'subspace_size' : 10,
                'max_recycle' : 10,
                'max_outer' : 10,
                'max_matvec' : 100, # this should be hit first
                'rel_tol'  : 1e-3,
                'abs_tol'  : 1e-5,
            }
            self.krylov = GCROT(
                self.primal_factory,
                optns=krylov_optns,
                dual_factory=self.dual_factory)
        else:
            krylov_optns = {
                'out_file' : 'kona_nested_krylov.dat',
                'subspace_size' : 20,
                'rel_tol'  : 1e-3,
                'abs_tol'  : 1e-5,
            }
            self.krylov = FGMRES(
                self.primal_factory,
                optns=krylov_optns,
                dual_factory=self.dual_factory)

    def linearize(self, at_kkt, at_state, at_adjoint):
        super(NestedKKTPreconditioner, self).linearize(
            at_kkt, at_state, at_adjoint)

        if self.use_gcrot:
            self.krylov.clear_subspace()

        if not self.nested_allocated:
            self.in_vec = ReducedKKTVector(
                CompositePrimalVector(
                    self.primal_factory.generate(),
                    self.dual_factory.generate()),
                self.dual_factory.generate())
            self.rhs_work = ReducedKKTVector(
                CompositePrimalVector(
                    self.primal_factory.generate(),
                    self.dual_factory.generate()),
                self.dual_factory.generate())
            self.nested_allocated = True

    def mult_kkt_approx(self, in_vec, out_vec):
        # modify the incoming vector
        self.in_vec.equals(in_vec)
        in_vec._dual.times(self.at_dual)
        # strip the slack variables out of the in/out vectors
        short_in = ReducedKKTVector(
            self.in_vec._primal._design, self.in_vec._dual)
        short_out = ReducedKKTVector(
            out_vec._primal._design, out_vec._dual)

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

    def solve(self, rhs, solution):
        # make sure we have a krylov solver
        if self.krylov is None:
            raise AttributeError('krylov solver not set')

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
        solution.equals(0.0)
        self.krylov.solve(
            self.mult_kkt_approx, self.rhs_work, solution, precond)

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

        # back out the actual dual solution
        solution._dual.times(self.at_dual)
