import numpy
import unittest

from kona import Optimizer
from kona.algorithms import FLECS_RSNK
from kona.examples import SimpleConstrained, ExponentialConstrained, Sellar

class InequalityFLECSRSNKTestCase(unittest.TestCase):

    def test_sellar(self):

        solver = Sellar()

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,
            'globalization' : 'trust',

            'trust' : {
                'init_radius' : 2.5,
                'max_radius' : 10.0,
                'min_radius' : 1e-4,
            },

            'penalty' : {
                'mu_init' : 1.0,
                'mu_pow' : 0.1,
                'mu_max' : 1e4,
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
                'subspace_size' : 20,
                'check_res'     : True,
                'rel_tol'       : 1e-8,
                'abs_tol'       : 1e-12,
            },
        }

        # algorithm = Verifier
        algorithm = FLECS_RSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = numpy.array([1.977, 0., 0.])
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 5e-2)

    def test_exponential_constrained(self):

        feasible = True
        if feasible:
            init_x = [1., 1.]
        else:
            init_x = [-1., -1.]

        solver = ExponentialConstrained(init_x=init_x)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 30,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,

            'trust' : {
                'init_radius' : 5.0,
                'max_radius' : 100.0,
                'min_radius' : 1e-4,
            },

            'penalty' : {
                'mu_init' : 10.0,
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
                'rel_tol'       : 1e-8,
                'abs_tol'       : 1e-12,
            },
        }

        algorithm = FLECS_RSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = numpy.zeros(solver.num_primal)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

    def test_with_simple_constrained(self):

        feasible = False
        if feasible:
            init_x = [0.51, 0.52, 0.53]
        else:
            init_x = [1.51, 1.52, 1.53]

        solver = SimpleConstrained(init_x=init_x, ineq=True)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,
            'globalization' : 'trust',

            'trust' : {
                'init_radius' : 10.0,
                'max_radius' : 100.0,
                'min_radius' : 1e-4,
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
                'rel_tol'       : 1e-8,
                'abs_tol'       : 1e-12,
            },
        }

        algorithm = FLECS_RSNK
        # algorithm = Verifier
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = -1.*numpy.ones(solver.num_primal)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-3)

if __name__ == "__main__":
    unittest.main()
