import unittest
import numpy

from kona.linalg.memory import KonaMemory
from kona.algorithms import EqualityConstrainedRSNK
from kona.examples import SimpleConstrained, Constrained2x2

class EqualityConstrainedRSNKTestCase(unittest.TestCase):

    def test_with_simple_constrained(self):

        solver = SimpleConstrained()
        # solver = Constrained2x2()
        km = KonaMemory(solver)

        optns = {
            'max_iter' : 30,
            'primal_tol' : 1e-6,
            'constraint_tol' : 1e-6,

            'trust' : {
                'init_radius' : 0.5,
                'max_radius' : 1.0,
            },

            'aug_lag' : {
                'mu_init' : 1.0,
                'mu_pow' : 1.0,
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
                'max_iter'      : 3,
                'rel_tol'       : 0.0095,
                'check_res'     : True,
            },
        }

        algorithm = EqualityConstrainedRSNK(
            km.primal_factory, km.state_factory, km.dual_factory, optns)
        km.allocate_memory()
        algorithm.solve()

        print solver.curr_design

        expected = -1.*numpy.ones(solver.num_primal)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-5)

if __name__ == "__main__":
    unittest.main()
