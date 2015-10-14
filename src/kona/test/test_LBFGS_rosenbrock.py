import unittest
import numpy

from kona.linalg.memory import KonaMemory
from kona.algorithms.util.linesearch import BackTracking, StrongWolfe
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.examples.rosenbrock import Rosenbrock
from kona.algorithms import ReducedSpaceQuasiNewton

class RosenbrockLBFGSTestCase(unittest.TestCase):

    def test_LBFGS_with_StrongWolfe(self):

        num_design = 2
        solver = Rosenbrock(num_design)
        km = KonaMemory(solver)

        optns = {
            'max_iter' : 200,
            'primal_tol' : 1e-12,
            'globalization' : {
                'type' : StrongWolfe,
            },
            'quasi_newton' : {
                'type' : LimitedMemoryBFGS
            },
        }
        rsqn = ReducedSpaceQuasiNewton(
            km.primal_factory, km.state_factory, optns)
        km.allocate_memory()
        rsqn.solve()

        expected = numpy.ones(num_design)
        diff = max(abs(solver.curr_design - expected))
        self.assertTrue(diff < 1.e-5)

    def test_LBFGS_with_BackTracking(self):
        num_design = 2
        solver = Rosenbrock(num_design)
        km = KonaMemory(solver)

        optns = {
            'max_iter' : 200,
            'primal_tol' : 1e-12,
            'globalization' : {
                'type' : BackTracking,
            },
            'quasi_newton' : {
                'type' : LimitedMemoryBFGS
            },
        }
        rsqn = ReducedSpaceQuasiNewton(
            km.primal_factory, km.state_factory, optns)
        km.allocate_memory()
        rsqn.solve()

        expected = numpy.ones(num_design)
        diff = max(abs(solver.curr_design - expected))
        self.assertTrue(diff < 1.e-5)

if __name__ == "__main__":
    unittest.main()
