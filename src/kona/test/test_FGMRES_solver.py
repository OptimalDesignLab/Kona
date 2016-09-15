import unittest

import numpy

from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.matrices.common import IdentityMatrix
from kona.user import UserSolver
from kona.linalg.memory import KonaMemory

class FLECSSolverTestCase(unittest.TestCase):

    def setUp(self):
        solver = UserSolver(4,0,0,0)
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.pf.request_num_vectors(7)
        optns = {
            'max_iter' : 30,
            'rel_tol' : 1e-10,
        }
        self.krylov = FGMRES(self.pf, optns)
        self.km.allocate_memory()

        self.x = self.pf.generate()
        self.b = self.pf.generate()
        self.b.equals(1.)
        self.b.base.data[0] = 2.0
        self.A = numpy.array([[4, 3, 2, 1],
                              [3, 4, 3, 2],
                              [2, 3, 4, 3],
                              [1, 2, 3, 4]])

        self.precond = IdentityMatrix()

    def mat_vec(self, in_vec, out_vec):
        in_data = in_vec.base.data.copy()
        out_data = self.A.dot(in_data)
        out_vec.base.data[:] = out_data[:]

    def test_solve(self):
        # reset the solution vector
        self.x.equals(0)
        # solve the system with FGMRES
        self.krylov.solve(self.mat_vec, self.b, self.x, self.precond.product)
        # calculate expected result
        expected = numpy.linalg.solve(self.A, self.b.base.data)
        # compare actual result to expected
        diff = abs(self.x.base.data - expected)
        diff = max(diff)
        self.assertTrue(diff < 1.e-6)

    def test_solve_underdetermined(self):
        # try solving a consistent underdetermined problem
        self.A = numpy.array([[4, 3, 2, 1],
                              [3, 4, 3, 2],
                              [2, 3, 4, 3],
                              [0, 0, 0, 0]])
        self.b.base.data[3] = 0.0
        # reset the solution vector
        self.x.equals(0)
        # solve the system with FGMRES
        iters, beta, = self.krylov.solve(self.mat_vec, self.b, self.x,
                                         self.precond.product)
        self.assertTrue(iters == 3)

    def test_solve_overdetermined(self):
        # try solving an overdetermined problem; must be a symmetric matrix
        self.A = numpy.array([[4, 3, 2, 0],
                              [3, 4, 3, 0],
                              [2, 3, 4, 0],
                              [0, 0, 0, 0]])
        self.b.equals(1.)
        self.krylov.check_LSgrad = True
        # reset the solution vector
        self.x.equals(0)
        # solve the system with FGMRES
        iters, beta, = self.krylov.solve(self.mat_vec, self.b, self.x,
                                         self.precond.product)
        
        


if __name__ == "__main__":

    unittest.main()
