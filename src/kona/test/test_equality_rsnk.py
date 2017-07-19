import numpy as np
import unittest

from kona import Optimizer
from kona.algorithms import ConstrainedRSNK
from kona.examples import ExponentialConstrained

class ConstrainedRSNKTestCase(unittest.TestCase):

    def test_with_exponential_constrained(self):
        '''ConstrainedRSNK test with ExponentialConstrained problem'''
        solver = ExponentialConstrained()

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,
            'globalization' : 'filter',
        
            'trust' : {
                'init_radius' : 1.0,
                'max_radius' : 4.0,
                'min_radius' : 1e-3,
            },
        
            'penalty' : {
                'mu_init' : 10.0,
                'mu_pow' : 1.0,
                'mu_max' : 1e5,
            },
        
            'rsnk' : {
                'precond'       : None,
                # rsnk algorithm settings
                'dynamic_tol'   : False,
                'nu'            : 0.95,
                # reduced KKT matrix settings
                'product_fac'   : 0.001,
                'lambda'        : 0.0,
                'scale'         : 1.0,
                'grad_scale'    : 1.0,
                'feas_scale'    : 1.0,
                # FLECS solver settings
                'krylov_file'   : 'kona_krylov.dat',
                'subspace_size' : 10,
                'check_res'     : True,
                'rel_tol'       : 0.005,
            }
        }

        algorithm = ConstrainedRSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()
        
        expected = np.zeros(solver.num_design)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()
