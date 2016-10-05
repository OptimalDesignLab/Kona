import numpy as np
import unittest

from kona import Optimizer
from kona.algorithms import PredictorCorrectorCnstr
from kona.examples import SphereConstrained

class PredictorCorrectorCnstrTestCase(unittest.TestCase):

    def test_with_simple_constrained(self):

        feasible = False
        if feasible:
            init_x = [0.51, 0.52, 0.53]
        else:
            init_x = [1.51, 1.52, 1.53]

        solver = SphereConstrained(init_x=init_x, ineq=False)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 100,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,

            'homotopy' : {
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
                # FLECS solver settings
                'krylov_file'   : 'kona_krylov.dat',
                'subspace_size' : 10,
                'rel_tol'       : 0.0095,
            },
        }

        algorithm = PredictorCorrectorCnstr
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = -1.*np.ones(solver.num_design)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()
