import unittest
import numpy as np
import sys

from kona import Optimizer
from kona.algorithms import PredictorCorrector
from kona.examples import Rosenbrock, Spiral

class PredictorCorrectorTestCase(unittest.TestCase):

    def test_param_cont_with_Rosenbrock(self):

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

    def test_RSNK_with_Spiral(self):

        solver = Spiral()

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-8,

            'homotopy' : {
                'mu' : 0.0,
                'inner_tol' : 1e-2,
                'inner_maxiter' : 50,
                'nominal_dist' : 10.0,
                'nominal_angle' : 10.0*np.pi/180.,
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
                'rel_tol'       : 1e-7,
            },

            'verify' : {
                'primal_vec'    : True,
                'state_vec'     : True,
                'dual_vec'      : False,
                'gradients'     : True,
                'pde_jac'       : True,
                'cnstr_jac'     : False,
                'red_grad'      : True,
                'lin_solve'     : True,
                'out_file'      : sys.stdout,
            },
        }

        algorithm = PredictorCorrector
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        diff = abs(solver.curr_design)
        self.assertTrue(max(diff) < 1e-2)

if __name__ == "__main__":
    unittest.main()
