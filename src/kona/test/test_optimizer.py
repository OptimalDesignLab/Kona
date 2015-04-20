import unittest
import numpy

from kona import Optimizer
from kona.examples import Rosenbrock
from kona.algorithms import ReducedSpaceQuasiNewton

class OptimizerTestCase(unittest.TestCase):

    def test_rosenbrock(self):
        num_design = 2
        solver = Rosenbrock(num_design)
        optns = {
            'primal_tol' : 1e-12,
        }
        optimizer = Optimizer(solver, ReducedSpaceQuasiNewton, optns)
        optimizer.solve()

        expected = numpy.ones(num_design)
        diff = max(abs(solver.curr_design - expected))
        self.assertTrue(diff < 1.e-5)

if __name__ == "__main__":
    unittest.main()
