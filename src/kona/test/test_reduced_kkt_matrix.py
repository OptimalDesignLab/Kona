import unittest

import numpy as np

from kona.examples import SimpleConstrained
from kona.linalg.memory import KonaMemory
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.vectors.composite import ReducedKKTVector, DesignSlackComposite


class ReducedKKTMatrixTestCase(unittest.TestCase):
    '''Test case for the Reduced Hessian approximation matrix.'''

    def _generate_KKT_vector(self):
        primal = self.pf.generate()
        slack = self.df.generate()
        dual = self.df.generate()
        return ReducedKKTVector(DesignSlackComposite(primal, slack), dual)

    def assertRelError(self, vec1, vec2, atol=1e-15):
        self.assertTrue(np.linalg.norm(vec1 - vec2) < atol)

    def test_equality_constrained_product(self):
        solver = SimpleConstrained(ineq=False)
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.dual_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(15)

        self.KKT_matrix = ReducedKKTMatrix([self.pf, self.sf, self.df])

        km.allocate_memory()

        # get memory
        primal_work = self.pf.generate()
        state = self.sf.generate()
        adjoint = self.sf.generate()
        state_work = self.sf.generate()
        dual_work = self.df.generate()
        X = self._generate_KKT_vector()
        dLdX = self._generate_KKT_vector()
        dLdX_pert = self._generate_KKT_vector()
        in_vec = self._generate_KKT_vector()
        out_vec = self._generate_KKT_vector()

        in_vec._primal.equals(2.0)
        in_vec._dual.equals(2.0)

        X._primal._design._data.data[0] = 0.51
        X._primal._design._data.data[1] = 0.52
        X._primal._design._data.data[2] = 0.53
        X._dual.equals(1.)
        state.equals_primal_solution(X._primal._design)
        adjoint.equals_adjoint_solution(X._primal._design, state, state_work)
        dLdX.equals_KKT_conditions(X, state, adjoint, primal_work, dual_work)

        epsilon_fd = 1e-6
        X.equals_ax_p_by(1.0, X, epsilon_fd, in_vec)
        state.equals_primal_solution(X._primal._design)
        adjoint.equals_adjoint_solution(X._primal._design, state, state_work)
        dLdX_pert.equals_KKT_conditions(
            X, state, adjoint, primal_work, dual_work)

        dLdX_pert.minus(dLdX)
        dLdX_pert.divide_by(epsilon_fd)

        X._primal._design._data.data[0] = 0.51
        X._primal._design._data.data[1] = 0.52
        X._primal._design._data.data[2] = 0.53
        X._dual.equals(1.)
        state.equals_primal_solution(X._primal._design)
        adjoint.equals_adjoint_solution(X._primal._design, state, state_work)
        self.KKT_matrix.linearize(X, state, adjoint)
        self.KKT_matrix.product(in_vec, out_vec)

        print '----------------------'
        print 'Equality Constraints'
        print '----------------------'
        print 'FD product:'
        print dLdX_pert._primal._design._data.data
        print dLdX_pert._primal._slack._data.data
        print dLdX_pert._dual._data.data
        print 'Analytical product:'
        print out_vec._primal._design._data.data
        print out_vec._primal._slack._data.data
        print out_vec._dual._data.data
        print '----------------------'

        dLdX.equals_ax_p_by(1.0, dLdX_pert, -1.0, out_vec)
        diff_norm = dLdX.norm2

        self.assertTrue(diff_norm <= 1e-3)

    def test_inequality_constrained_product(self):
        solver = SimpleConstrained(ineq=True)
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.dual_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(15)

        self.KKT_matrix = ReducedKKTMatrix([self.pf, self.sf, self.df])

        km.allocate_memory()

        # get memory
        primal_work = self.pf.generate()
        state = self.sf.generate()
        adjoint = self.sf.generate()
        state_work = self.sf.generate()
        dual_work = self.df.generate()
        X = self._generate_KKT_vector()
        dLdX = self._generate_KKT_vector()
        dLdX_pert = self._generate_KKT_vector()
        in_vec = self._generate_KKT_vector()
        out_vec = self._generate_KKT_vector()

        in_vec.equals(2.0)

        X._primal._design._data.data[0] = 1.51
        X._primal._design._data.data[1] = 1.52
        X._primal._design._data.data[2] = 1.53
        X._primal._slack.equals(2.)
        X._dual.equals(1.)
        state.equals_primal_solution(X._primal._design)
        adjoint.equals_adjoint_solution(X._primal._design, state, state_work)
        dLdX.equals_KKT_conditions(X, state, adjoint, primal_work, dual_work)

        epsilon_fd = 1e-6
        X.equals_ax_p_by(1.0, X, epsilon_fd, in_vec)
        state.equals_primal_solution(X._primal._design)
        adjoint.equals_adjoint_solution(X._primal._design, state, state_work)
        dLdX_pert.equals_KKT_conditions(
            X, state, adjoint, primal_work, dual_work)

        dLdX_pert.minus(dLdX)
        dLdX_pert.divide_by(epsilon_fd)

        X._primal._design._data.data[0] = 1.51
        X._primal._design._data.data[1] = 1.52
        X._primal._design._data.data[2] = 1.53
        X._primal._slack.equals(2.)
        X._dual.equals(1.)
        state.equals_primal_solution(X._primal._design)
        adjoint.equals_adjoint_solution(X._primal._design, state, state_work)
        self.KKT_matrix.linearize(X, state, adjoint)
        self.KKT_matrix.product(in_vec, out_vec)

        print '----------------------'
        print 'Inequality Constraints'
        print '----------------------'
        print 'FD product:'
        print dLdX_pert._primal._design._data.data
        print dLdX_pert._primal._slack._data.data
        print dLdX_pert._dual._data.data
        print 'Analytical product:'
        print out_vec._primal._design._data.data
        print out_vec._primal._slack._data.data
        print out_vec._dual._data.data
        print '----------------------'

        dLdX.equals_ax_p_by(1.0, dLdX_pert, -1.0, out_vec)
        diff_norm = dLdX.norm2

        self.assertTrue(diff_norm <= 1e-3)

if __name__ == "__main__":
    unittest.main()
