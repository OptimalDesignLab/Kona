import unittest
import numpy

from kona.linalg.memory import KonaMemory
from kona.algorithms.util.linesearch import BackTracking, StrongWolfe
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.examples import Rosenbrock
from kona.user import UserSolver
from kona.algorithms import ReducedSpaceQuasiNewton
from kona.options import BadKonaOption

class ReducedSpaceQuasiNewtonTestCase(unittest.TestCase):

    def setUp(self):
        solver = UserSolver(1, 1)
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory

        self.pf.request_num_vectors(1)
        self.sf.request_num_vectors(1)

        km.allocate_memory()

    def test_no_quasi_newton(self):
        '''ReducedSpaceQuasiNewton error messages'''
        optns = {'quasi_newton': {'type': None}}
        try:
            ReducedSpaceQuasiNewton(self.pf, self.sf, None, None, optns)
        except BadKonaOption as err:
            self.assertEqual(
                str(err),
                "Invalid Kona option: optns['quasi_newton']['type'] = None")

        optns = {'quasi_newton': None}
        try:
            ReducedSpaceQuasiNewton(self.pf, self.sf, None, None, optns)
        except BadKonaOption as err:
            self.assertEqual(
                str(err),
                "Invalid Kona option: optns['quasi_newton']['type'] = None")

        optns = {'quasi_newton': {'type': 25}}
        try:
            ReducedSpaceQuasiNewton(self.pf, self.sf, None, None, optns)
        except BadKonaOption as err:
            self.assertEqual(
                str(err),
                "Invalid Kona option: optns['quasi_newton']['type'] = 25")

        optns = {'quasi_newton': {'type': LimitedMemoryBFGS}}
        try:
            ReducedSpaceQuasiNewton(self.pf, self.sf, None, None, optns)
        except Exception:
            self.fail('No Error Expected')

        try:
            ReducedSpaceQuasiNewton(self.pf, self.sf, None, None)
        except Exception:
            self.fail('No Error Expected')

    def test_no_line_search(self):

        optns = {'quasi_newton': {'type': LimitedMemoryBFGS}}
        try:
            ReducedSpaceQuasiNewton(self.pf, self.sf, None, None, optns)
        except BadKonaOption as err:
            self.assertEqual(
                str(err),
                "Invalid Kona option: optns['globalization']['type'] = None")

    def test_LBFGS_with_BackTracking(self):
        '''ReducedSpaceQuasiNewton with BackTracking line search'''
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

    def test_LBFGS_with_StrongWolfe(self):
        '''ReducedSpaceQuasiNewton with StrongWolfe line search'''
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

if __name__ == "__main__":
    unittest.main()
