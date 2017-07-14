import numpy as np
import unittest

from kona import Optimizer
from kona.algorithms import ConstrainedRSNK
from kona.examples import SimpleIDF

class IDFSchurTestCase(unittest.TestCase):

    def test_with_precond_active(self):
        '''ReducedSchurPreconditioner optimization test with SimpleIDF problem'''
        solver = SimpleIDF(num_disc=5, init_x=1, approx_inv=False)
        
        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 5,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,
            'globalization' : 'trust',
        
            'trust' : {
                'init_radius' : 1.0,
                'max_radius' : 4.0,
                'min_radius' : 1e-3,
            },
        
            'penalty' : {
                'mu_init' : 100.0,
                'mu_pow' : 0.5,
                'mu_max' : 1e5,
            },
        
            'rsnk' : {
                'precond'       : 'idf_schur',
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
                'rel_tol'       : 0.0095,
            }
        }
        
        algorithm = ConstrainedRSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()
        
        error = np.linalg.norm(solver.curr_design)
        self.assertTrue(error < 1e-8)

if __name__ == "__main__":
    unittest.main()
