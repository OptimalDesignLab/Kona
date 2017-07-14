import unittest

import numpy

from kona.linalg.solvers.krylov import STCG
from kona.linalg.matrices.common import IdentityMatrix
from kona.user import UserSolver
from kona.linalg.memory import KonaMemory

class STCGSolverTestCase(unittest.TestCase):

    def setUp(self):
        solver = UserSolver(4,0,0,0)
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.pf.request_num_vectors(7)
        optns = {
            'max_iter' : 30,
            'rel_tol' : 1e-3,
        }
        self.krylov = STCG(self.pf, optns)
        self.km.allocate_memory()

        self.x = self.pf.generate()
        self.b = self.pf.generate()
        self.b.equals(1)
        self.A = numpy.array([[4, 3, 2, 1],
                              [3, 4, 3, 2],
                              [2, 3, 4, 3],
                              [1, 2, 3, 4]])

        self.precond = IdentityMatrix()

    def mat_vec(self, in_vec, out_vec):
        in_data = in_vec.base.data.copy()
        out_data = self.A.dot(in_data)
        out_vec.base.data[:] = out_data[:]

    def test_bad_radius(self):
        '''STCG error test for bad trust radius'''
        # reset the solution vector
        self.x.equals(0)
        # solve the system with CG
        self.krylov.radius = -1.
        try:
            self.krylov.solve(
                self.mat_vec, self.b, self.x, self.precond.product)
        except ValueError as err:
            self.assertEqual(str(err), 'radius must be postive')

    def test_radius_inactive(self):
        '''STCG solution test with inactive trust radius'''
        # reset the solution vector
        self.x.equals(0)
        # solve the system with CG
        self.krylov.radius = 1.0
        prec, active = self.krylov.solve(self.mat_vec, self.b, self.x,
                                         self.precond.product)
        # calculate expected result
        expected = numpy.array([.2, 0, 0, .2])
        # compare actual result to expected
        diff = abs(self.x.base.data - expected)
        diff = max(diff)
        self.assertTrue(diff < 1.e-6)
        self.assertTrue(not active)
        self.assertTrue(abs(prec - 0.2) <= 1e-12)

    def test_radius_active(self):
        '''STCG solution test with active trust radius'''
        # reset the solution vector
        self.x.equals(0)
        # solve the system with CG
        self.krylov.radius = 0.1
        prec, active = self.krylov.solve(self.mat_vec, self.b, self.x,
                                         self.precond.product)
        # compare actual result to expected
        exp_norm = self.krylov.radius
        actual_norm = self.x.norm2
        self.assertTrue(abs(actual_norm - exp_norm) <= 1e-5)
        self.assertTrue(active)
        self.assertTrue(abs(prec - 0.145) <= 1e-12)

if __name__ == "__main__":

    unittest.main()
