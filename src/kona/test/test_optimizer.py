import unittest
import numpy

import kona

class OptimizerTestCase(unittest.TestCase):

    def test_rosenbrock(self):
        num_design = 2
        solver = kona.examples.Rosenbrock(num_design)
        optns = {
            'primal_tol' : 1e-12,
        }
        algorithm = kona.algorithms.ReducedSpaceQuasiNewton
        optimizer = kona.Optimizer(solver, algorithm, optns)
        optimizer.solve()

        expected = numpy.ones(num_design)
        diff = max(abs(solver.curr_design - expected))
        self.assertTrue(diff < 1.e-5)

if __name__ == "__main__":
    unittest.main()
