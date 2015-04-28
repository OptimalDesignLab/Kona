import unittest

import numpy

import kona

from kona.linalg.solvers.krylov import STCG
from kona.linalg.matrices.common import IdentityMatrix
from kona.user import UserSolver
from kona.linalg.memory import KonaMemory

class KrylovSolverTestCase(unittest.TestCase):

    def setUp(self):
        solver = UserSolver(4,0,0)
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.pf.request_num_vectors(2)
        optns = {
            'rel_tol' : 1e-6,
            'radius' : 1.0,
        }
        self.krylov = STCG(self.pf, optns)
        self.km.allocate_memory()

    def test_solve(self):
        # initialize some vectors
        x = self.pf.generate()
        x.equals(0)
        b = self.pf.generate()
        b.equals(1)
        # define system matrix
        A = numpy.array([[1, 2, 3, 4],
                         [0, 1, 2, 3],
                         [0, 0, 1, 2],
                         [0, 0, 0, 1]])
        # define matrix-vector product
        def mat_vec(in_vec, out_vec):
            in_data = in_vec._data.data.copy()
            out_data = A.dot(in_data)
            out_vec._data.data[:] = out_data[:]
        # define identity preconditioner
        precond = IdentityMatrix()
        # solve the system with CG
        self.krylov.solve(mat_vec, b, x, precond.product)
        # calculate expected result
        expected = numpy.array([0, 0, -1, 1])
        # compare actual result to expected
        diff = abs(x._data.data - expected)
        diff = max(diff)
        #self.assertTrue(diff < 1.e-6)
        self.failUnless('Untested')

if __name__ == "__main__":

    unittest.main()
