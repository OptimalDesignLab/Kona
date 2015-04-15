import unittest
import numpy

from kona.linalg.memory import KonaMemory
from kona.algorithms.util.linesearch import BackTracking, StrongWolfe
from kona.linalg.matrices.lbfgs import LimitedMemoryBFGS
from kona.linalg.matrices.lsr1 import LimitedMemorySR1
from kona.examples.rosenbrock import Rosenbrock
from kona.algorithms.reduced_space_quasi_newton import ReducedSpaceQuasiNewton

class SolveRosenbrockTestCase(unittest.TestCase):

    def test_rosenbrock_opt(self):

        num_design = 2
        solver = Rosenbrock(num_design)
        km = KonaMemory(solver)

        optns = {
            'max_iter' : 200,
            'primal_tol' : 1e-12,
            'line_search' : {
                'type' : StrongWolfe,
            },
            'quasi_newton' : {
                'type' : LimitedMemoryBFGS
            },
        }
        rsqn = ReducedSpaceQuasiNewton(km.primal_factory, km.state_factory, optns)
        km.allocate_memory()
        rsqn.solve()

        expected = numpy.ones(num_design)
        diff = max(abs(solver.curr_design - expected))
        self.assertTrue(diff < 1.e-2)

if __name__ == "__main__":
    unittest.main()
