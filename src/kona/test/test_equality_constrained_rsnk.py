import numpy
import unittest

from kona.algorithms import ConstrainedRSNK
from kona.examples import SimpleConstrained
from kona.linalg.memory import KonaMemory

class EqualityConstrainedRSNKTestCase(unittest.TestCase):

    # def test_dummy(self):
    #     self.failUnless('Untested')

    def test_with_simple_constrained(self):

        solver = SimpleConstrained(ineq=False)
        km = KonaMemory(solver)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'primal_tol' : 1e-5,
            'constraint_tol' : 1e-5,

            'trust' : {
                'init_radius' : 1.0,
                'max_radius' : 10.0,
                'min_radius' : 1e-4,
            },

            'aug_lag' : {
                'mu_init' : 1.0,
                'mu_pow' : 0.5,
                'mu_max' : 1e5,
            },

            'reduced' : {
                'precond'       : None,
                'product_fac'   : 0.001,
                'lambda'        : 0.0,
                'scale'         : 0.0,
                'nu'            : 0.95,
                'dynamic_tol'   : False,
            },

            'krylov' : {
                'out_file'      : 'kona_krylov.dat',
                'max_iter'      : 10,
                'rel_tol'       : 0.00095,
                'check_res'     : True,
            },
        }

        algorithm = ConstrainedRSNK(
            km.primal_factory, km.state_factory, km.dual_factory, optns)
        km.allocate_memory()
        algorithm.solve()

        print solver.curr_design

        expected = -1.*numpy.ones(solver.num_primal)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()
