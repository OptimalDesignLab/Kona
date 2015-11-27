import unittest
import numpy

from kona.linalg.memory import KonaMemory
from kona.algorithms.util.linesearch import BackTracking, StrongWolfe
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.examples import Rosenbrock, Spiral
from kona.algorithms import ReducedSpaceQuasiNewton

class RosenbrockLBFGSTestCase(unittest.TestCase):

    def test_LBFGS_with_StrongWolfe(self):

        num_design = 2
        solver = Rosenbrock(num_design)
        km = KonaMemory(solver)

        optns = {
            'info_file' : 'kona_info.dat',
            'hist_file' : 'kona_hist.dat',
            'max_iter' : 200,
            'opt_tol' : 1e-12,

            'globalization' : 'linesearch',

            'linesearch' : {
                'type'          : StrongWolfe,
            },

            'quasi_newton' : {
                'type' : LimitedMemoryBFGS
            },
        }

        rsqn = ReducedSpaceQuasiNewton(
            km.primal_factory, km.state_factory, None, optns)
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
            'info_file' : 'kona_info.dat',
            'hist_file' : 'kona_hist.dat',
            'max_iter' : 200,
            'opt_tol' : 1e-12,

            'globalization' : 'linesearch',

            'linesearch' : {
                'type'     : BackTracking,
            },

            'quasi_newton' : {
                'type' : LimitedMemoryBFGS
            },
        }
        rsqn = ReducedSpaceQuasiNewton(
            km.primal_factory, km.state_factory, None, optns)
        km.allocate_memory()
        rsqn.solve()

        expected = numpy.ones(num_design)
        diff = max(abs(solver.curr_design - expected))
        self.assertTrue(diff < 1.e-5)

    def test_LBFGS_with_Spiral(self):
        solver = Spiral()
        km = KonaMemory(solver)

        optns = {
            'info_file' : 'kona_info.dat',
            'hist_file' : 'kona_hist.dat',
            'max_iter' : 200,
            'opt_tol'  : 1e-12,

            'globalization' : 'linesearch',

            'linesearch' : {
                'type'     : StrongWolfe,
            },

            'quasi_newton' : {
                'type' : LimitedMemoryBFGS
            },
        }

        rsqn = ReducedSpaceQuasiNewton(
            km.primal_factory, km.state_factory, None, optns)
        km.allocate_memory()
        rsqn.solve()

        # expected = numpy.ones(num_design)
        # diff = max(abs(solver.curr_design - expected))
        # self.assertTrue(diff < 1.e-5)
        self.failUnless('Untested')

if __name__ == "__main__":
    unittest.main()
