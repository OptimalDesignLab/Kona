import numpy
import unittest

from kona import Optimizer
from kona.algorithms import ConstrainedRSNK
from kona.examples import SimpleConstrained, ExponentialConstrained, Sellar

class InequalityConstrainedRSNKTestCase(unittest.TestCase):

    def test_sellar(self):

        solver = Sellar()

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'primal_tol' : 1e-5,
            'constraint_tol' : 1e-5,

            'trust' : {
                'init_radius' : 0.1,
                'max_radius' : 4.0,
                'min_radius' : 1e-4,
            },

            'aug_lag' : {
                'mu_init' : 1000.0,
                'mu_pow' : 0.0,
                'mu_max' : 1e5,
            },
        }

        # algorithm = Verifier
        algorithm = ConstrainedRSNK
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
            'primal_tol' : 1e-5,
            'constraint_tol' : 1e-5,

            'trust' : {
                'init_radius' : 1.0,
                'max_radius' : 10.0,
                'min_radius' : 1e-4,
            },

            'aug_lag' : {
                'mu_init' : 1.0,
                'mu_pow' : 0.5,
                'mu_max' : 1e5,
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
                'max_iter'      : 10,
                'rel_tol'       : 0.0095,
                'check_res'     : True,
            },
        }

        algorithm = ConstrainedRSNK
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
            'primal_tol' : 1e-5,
            'constraint_tol' : 1e-5,
            'globalization' : 'trust',
            # 'globalization' : None,

            'trust' : {
                'init_radius' : 0.5,
                'max_radius' : 20.0,
                'min_radius' : 1e-4,
            },

            'aug_lag' : {
                'mu_init' : 0.1,
                'mu_pow' : 0.1,
                'mu_max' : 1e5,
            },
        }

        algorithm = ConstrainedRSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = -1.*numpy.ones(solver.num_primal)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()
