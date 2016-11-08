import numpy as np
import sys
import unittest

from kona import Optimizer
from kona.algorithms import PredictorCorrectorCnstr, FLECS_RSNK, Verifier
from kona.examples import SimpleIDF

class IDFSchurTestCase(unittest.TestCase):

    def test_with_simple_idf(self):

        solver = SimpleIDF(num_disc=5, init_x=1, approx_inv=False)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 100,
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

            'homotopy': {
                'inner_tol': 1e-2,
                'inner_maxiter': 20,
                'nominal_dist': 10.0,
                'nominal_angle': 20.0 * np.pi / 180.,
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
            },

            'verify': {
                'primal_vec': True,
                'state_vec': True,
                'dual_vec_eq': True,
                'dual_vec_in': False,
                'gradients': True,
                'pde_jac': True,
                'cnstr_jac_eq': True,
                'cnstr_jac_in': False,
                'red_grad': True,
                'lin_solve': True,
                'out_file': sys.stdout,
            },
        }

        # algorithm = FLECS_RSNK
        algorithm = PredictorCorrectorCnstr
        # algorithm = Verifier
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        error = abs(solver.curr_design[0])
        self.assertTrue(error < 1e-4)

if __name__ == "__main__":
    unittest.main()
