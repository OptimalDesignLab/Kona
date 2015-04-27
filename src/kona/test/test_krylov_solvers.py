import unittest

import numpy

import os, sys
kona_path = os.path.abspath('../..')
sys.path.append(kona_path)

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
        self.pf.request_num_vectors(7)
        optns = {
            'max_iter' : 30,
            'rel_tol' : 1e-3,
            'radius' : 1,
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
        A = numpy.array([[4, 3, 2, 1],
                         [3, 4, 3, 2],
                         [2, 3, 4, 3],
                         [1, 2, 3, 4]])
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
        expected = numpy.array([.2, 0, 0, .2])
        # compare actual result to expected
        diff = abs(x._data.data - expected)
        diff = max(diff)
        #self.assertTrue(diff < 1.e-6)
        self.failUnless('Untested')


if __name__ == "__main__":

    unittest.main()
