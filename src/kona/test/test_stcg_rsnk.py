import unittest
import numpy
import sys

from kona import Optimizer
from kona.algorithms import STCG_RSNK, Verifier
from kona.examples import Rosenbrock, Spiral

class STCGRSNKTestCase(unittest.TestCase):

    def test_RSNK_with_Rosenbrock(self):

        num_design = 2
        solver = Rosenbrock(num_design)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-12,

            'trust' : {
                'init_radius' : 0.5,
                'max_radius' : 2.0,
                'min_radius' : 1e-4,
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
        }

        algorithm = STCG_RSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        diff = abs(solver.curr_design - numpy.ones(num_design))
        self.assertTrue(max(diff) < 1e-5)

    def test_RSNK_with_Spiral(self):

        solver = Spiral()

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-8,

            'trust' : {
                'init_radius' : 10.0,
                'max_radius' : 100.0,
                'min_radius' : 1e-4,
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

        algorithm = STCG_RSNK
        # algorithm = Verifier
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        diff = abs(solver.curr_design)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()
