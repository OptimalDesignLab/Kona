import unittest

import numpy as np

from kona.examples import SphereConstrained
from kona.linalg.memory import KonaMemory
from kona.linalg.matrices.common import dCdU, dRdU
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.vectors.composite import ReducedKKTVector


class ReducedKKTMatrixTestCase(unittest.TestCase):
    '''Test case for the Reduced Hessian approximation matrix.'''

    def _generate_KKT_vector(self):
        primal = self.pf.generate()
        dual = self.df.generate()
        return ReducedKKTVector(primal, dual)

    def assertRelError(self, vec1, vec2, atol=1e-15):
        self.assertTrue(np.linalg.norm(vec1 - vec2) < atol)

    def test_equality_constrained_product(self):
        solver = SphereConstrained(ineq=False)
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.eq_factory

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

        in_vec.primal.equals(2.0)
        in_vec.dual.equals(2.0)

        X.primal.base.data[0] = 0.51
        X.primal.base.data[1] = 0.52
        X.primal.base.data[2] = 0.53
        X.dual.equals(-1.0)
        state.equals_primal_solution(X.primal)
        state_work.equals_objective_partial(X.primal, state)
        dCdU(X.primal, state).T.product(X.dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X.primal, state).T.solve(state_work, adjoint)
        dLdX.equals_KKT_conditions(X, state, adjoint)

        epsilon_fd = 1e-6
        X.equals_ax_p_by(1.0, X, epsilon_fd, in_vec)
        state.equals_primal_solution(X.primal)
        state_work.equals_objective_partial(X.primal, state)
        dCdU(X.primal, state).T.product(X.dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X.primal, state).T.solve(state_work, adjoint)
        dLdX_pert.equals_KKT_conditions(X, state, adjoint)

        dLdX_pert.minus(dLdX)
        dLdX_pert.divide_by(epsilon_fd)

        X.primal.base.data[0] = 0.51
        X.primal.base.data[1] = 0.52
        X.primal.base.data[2] = 0.53
        X.dual.equals(-1.)
        state.equals_primal_solution(X.primal)
        adjoint.equals_objective_adjoint(X.primal, state, state_work)
        self.KKT_matrix.linearize(X, state, adjoint)
        self.KKT_matrix.product(in_vec, out_vec)

        print '----------------------'
        print 'Equality Constraints'
        print '----------------------'
        print 'FD product:'
        print dLdX_pert.primal.base.data
        print dLdX_pert.dual.base.data
        print 'Analytical product:'
        print out_vec.primal.base.data
        print out_vec.dual.base.data
        print '----------------------'

        dLdX.equals_ax_p_by(1.0, dLdX_pert, -1.0, out_vec)
        diff_norm = dLdX.norm2

        self.assertTrue(diff_norm <= 1e-3)

if __name__ == "__main__":
    unittest.main()
