import numpy
import unittest

from kona import Optimizer
from kona.algorithms import CompositeStep
from kona.examples import SimpleConstrained

class EqualityCompositeStepTestCase(unittest.TestCase):

    # def test_dummy(self):
    #     self.failUnless('Untested')

    def test_with_simple_constrained(self):

        solver = SimpleConstrained(
            init_x=[1.51, -1.52, 1.53],
            ineq=False)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,
            'globalization' : 'trust',

            'trust' : {
                'init_radius' : 0.1,
                'max_radius' : 1.0,
                'min_radius' : 1e-4,
            },

            'penalty' : {
                'mu_init' : 0.1,
                'mu_pow' : 0.1,
                'mu_max' : 1e5,
            },

            'composite-step' : {
                'normal-step' : {
                    'precond'     : None,
                    'lanczos_size': 1,
                    'out_file'    : 'kona_normal_krylov.dat',
                    'max_inner'   : 10,
                    'max_outer'   : 10,
                    'max_recycle' : 10,
                    'max_matvec'  : 100,
                    'check_res'   : True,
                    'rel_tol'     : 1e-8,
                },
                'tangent-step' : {
                    'out_file'    : 'kona_tangent_krylov.dat',
                    'max_iter'    : 50,
                    'check_res'   : True,
                    'rel_tol'     : 1e-8,
                }
            },
        }

        algorithm = CompositeStep
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = -1.*numpy.ones(solver.num_primal)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()
