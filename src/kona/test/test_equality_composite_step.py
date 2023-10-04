import numpy
import unittest

from kona import Optimizer
from kona.algorithms import CompositeStepRSNK
from kona.examples import SphereConstrained

class EqualityCompositeStepTestCase(unittest.TestCase):

    def test_with_simple_constrained(self):
        '''CompositeStepRSNK tested with Sphere problem'''
        init_x = [0.51, 0.52, 0.53]
        solver = SphereConstrained(init_x=init_x, ineq=False)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,
            'globalization' : 'linesearch',

            'trust' : {
                'init_radius' : 4.0,
                'max_radius' : 1.0,
                'min_radius' : 1e-4,
            },

            'penalty' : {
                'mu_init' : 1.0,
                'mu_pow' : 1e-8,
            },

            'composite-step' : {
                'normal-step' : {
                    'precond'     : None,
                    'lanczos_size': 1,
                    'use_gcrot'   : False,
                    'out_file'    : 'kona_normal_krylov.dat',
                    'max_inner'   : 10,
                    'max_outer'   : 10,
                    'max_recycle' : 10,
                    'max_matvec'  : 100,
                    'check_res'   : True,
                    'rel_tol'     : 1e-5,
                    'abs_tol'     : 1e-8,
                },
                'tangent-step' : {
                    'out_file'    : 'kona_tangent_krylov.dat',
                    'max_iter'    : 50,
                    'check_res'   : True,
                    'rel_tol'     : 1e-5,
                    'abs_tol'     : 1e-8,
                }
            },
        }

        algorithm = CompositeStepRSNK
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print(solver.curr_design)

        expected = -1.*numpy.ones(solver.num_design)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

if __name__ == "__main__":
    unittest.main()