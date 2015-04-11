import unittest
import numpy

from kona.algorithms.util.linesearch import StrongWolfe
from kona.examples.rosenbrock import Rosenbrock
from kona.algorithms.reduced_space_quasi_newton import ReducedSpaceQuasiNewton
from kona.linalg.memory import KonaMemory

class SolveRosenbrockTestCase(unittest.TestCase):

    def test_rosenbrock_opt(self):

        solver = Rosenbrock()
        km = KonaMemory(solver)

        rsqn = ReducedSpaceQuasiNewton(km.primal_factory, km.state_factory)
        km.allocate_memory()
        rsqn.solve()

        print solver.curr_design
        self.fail()

if __name__ == "__main__":
    unittest.main()
