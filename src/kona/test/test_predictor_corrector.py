import unittest
import numpy as np

from kona import Optimizer
from kona.algorithms import PredictorCorrector
from kona.examples import Rosenbrock

class PredictorCorrectorTestCase(unittest.TestCase):

    def test_param_cont_with_Rosenbrock(self):
        '''PredictorCorrector solution on Rosenbrock'''
        ndv = 2
        solver = Rosenbrock(ndv)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-8,

            'homotopy' : {
                'lambda' : 0.0,
                'inner_tol' : 1e-2,
                'inner_maxiter' : 50,
                'nominal_dist' : 1.0,
                'nominal_angle' : 7.0*np.pi/180.,
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
                # krylov solver settings
                'krylov_file'   : 'kona_krylov.dat',
                'subspace_size' : 10,
                'check_res'     : True,
                'rel_tol'       : 1e-5,
            },
        }

        algorithm = PredictorCorrector
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        diff = abs(solver.curr_design - np.ones(ndv))
        self.assertTrue(max(diff) < 1e-5)

if __name__ == "__main__":
    unittest.main()
