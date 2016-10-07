import numpy
import unittest

from kona import Optimizer
from kona.algorithms import FLECS_RSNK
from kona.examples import SphereConstrained

class EqualityFLECSRSNKTestCase(unittest.TestCase):

    def test_with_simple_constrained(self):

        init_x = [-2.51, -2.52, -2.53]

        solver = SphereConstrained(init_x=init_x, ineq=False)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,
            'globalization' : 'trust',

            'trust' : {
                'init_radius' : 1.0,
                'max_radius' : 4.0,
                'min_radius' : 1e-3,
            },

            'penalty' : {
                'mu_init' : 0.1,
                'mu_pow' : 0.1,
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
                'rel_tol'       : 0.00095,
            },
        }

        algorithm = FLECS_RSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = -1.*numpy.ones(solver.num_design)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()
