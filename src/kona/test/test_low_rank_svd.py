import unittest

from kona.examples import Sellar
from kona.linalg.memory import KonaMemory
from kona.linalg.matrices.common import dCdU, dRdU
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import ConstrainedHessian
from kona.linalg.matrices.preconds import LowRankSVD
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector


class LowRankSVDTestCase(unittest.TestCase):

    def _generate_KKT_vector(self):
        design = self.pf.generate()
        slack = self.df.generate()
        primal = CompositePrimalVector(design, slack)
        dual = self.df.generate()
        return ReducedKKTVector(primal, dual)

    def test_with_rectangular(self):
        solver = Sellar()
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.dual_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(15)

        self.A = TotalConstraintJacobian([self.pf, self.sf, self.df])

        def fwd_mat_vec(in_vec, out_vec):
            self.A.product(in_vec, out_vec)

        def rev_mat_vec(in_vec, out_vec):
            self.A.T.product(in_vec, out_vec)

        svd_optns = {'lanczos_size': 3}
        self.svd = LowRankSVD(
            fwd_mat_vec, self.pf, rev_mat_vec, self.df, svd_optns)

        km.allocate_memory()

        X = self._generate_KKT_vector()
        in_vec = self._generate_KKT_vector()
        out_vec_exact = self._generate_KKT_vector()
        out_vec_approx = self._generate_KKT_vector()
        state = self.sf.generate()

        in_vec.equals(2.0)
        X.equals_init_guess()
        state.equals_primal_solution(X._primal._design)
        self.A.linearize(X._primal._design, state)
        self.svd.linearize()

        self.A.product(in_vec._primal._design, out_vec_exact._dual)
        self.svd.approx_fwd_prod(in_vec._primal._design, out_vec_approx._dual)

        print 'Constraint Jacobian test:'

        print 'Exact fwd product  =', out_vec_exact._dual.norm2
        print 'Approx fwd product =', out_vec_approx._dual.norm2

        out_vec_approx._dual.minus(out_vec_exact._dual)
        abs_error = out_vec_approx._dual.norm2
        rel_error = abs_error/out_vec_exact._dual.norm2

        print 'Abs error norm     =', abs_error
        print 'Rel error norm     =', rel_error

        self.assertTrue(rel_error <= 1e-8)

        self.A.T.product(in_vec._dual, out_vec_exact._primal._design)
        self.svd.approx_rev_prod(in_vec._dual, out_vec_approx._primal._design)

        print 'Exact rev product  =', out_vec_exact._primal._design.norm2
        print 'Approx fwd product =', out_vec_approx._primal._design.norm2

        out_vec_approx._primal._design.minus(out_vec_exact._primal._design)
        abs_error = out_vec_approx._primal._design.norm2
        rel_error = abs_error/out_vec_exact._primal._design.norm2

        print 'Abs error norm     =', abs_error
        print 'Rel error norm     =', rel_error

        self.assertTrue(rel_error <= 1e-8)

    def test_with_square(self):
        solver = Sellar()
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.dual_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(15)

        self.W = ConstrainedHessian([self.pf, self.sf, self.df])

        def mat_vec(in_vec, out_vec):
            self.W.product(in_vec, out_vec)

        svd_optns = {'lanczos_size': 3}
        self.svd = LowRankSVD(
            mat_vec, self.pf, optns=svd_optns)

        km.allocate_memory()

        X = self._generate_KKT_vector()
        in_vec = self._generate_KKT_vector()
        out_vec_exact = self._generate_KKT_vector()
        out_vec_approx = self._generate_KKT_vector()
        state = self.sf.generate()
        state_work = self.sf.generate()
        adjoint = self.sf.generate()

        in_vec.equals(2.0)
        X.equals_init_guess()
        state.equals_primal_solution(X._primal._design)
        state_work.equals_objective_partial(X._primal._design, state)
        dCdU(X._primal._design, state).T.product(X._dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X._primal._design, state).T.solve(state_work, adjoint)
        self.W.linearize(X._primal._design, X._dual, state, adjoint)
        self.svd.linearize()

        self.W.product(in_vec._primal._design, out_vec_exact._primal._design)
        self.svd.approx_fwd_prod(
            in_vec._primal._design, out_vec_approx._primal._design)

        print 'Hessian test:'
        print 'Exact product  =', out_vec_exact._primal._design.norm2
        print 'Approx product =', out_vec_approx._primal._design.norm2

        out_vec_approx._primal._design.minus(out_vec_exact._primal._design)
        abs_error = out_vec_approx._primal._design.norm2
        rel_error = abs_error/out_vec_exact._primal._design.norm2

        print 'Abs error norm =', abs_error
        print 'Rel error norm =', rel_error

        self.assertTrue(rel_error <= 1e-8)

if __name__ == "__main__":
    unittest.main()
