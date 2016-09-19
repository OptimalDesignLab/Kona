import numpy
import unittest
import sys

from kona import Optimizer
from kona.algorithms import CompositeStepRSNK, Verifier
from kona.examples import SphereConstrained, ExponentialConstrained, Sellar

class InequalityCompositeStepTestCase(unittest.TestCase):

    def test_dummy(self):
        untested = True
        self.assertTrue(untested)

    # def test_sellar(self):
    #
    #     solver = Sellar()
    #
    #     optns = {
    #         'info_file' : 'kona_info.dat',
    #         'max_iter' : 30,
    #         'opt_tol' : 1e-5,
    #         'feas_tol' : 1e-5,
    #
    #         'globalization' : 'trust',
    #
    #         'trust' : {
    #             'init_radius' : 5.0,
    #             'max_radius' : 100.0,
    #             'min_radius' : 1e-4,
    #         },
    #
    #         'penalty' : {
    #             'mu_init' : 1.0,
    #             'mu_pow' : 1e-8,
    #         },
    #
    #         'composite-step' : {
    #             'normal-step' : {
    #                 'precond'       : None,
    #                 'lanczos_size'  : 3,
    #                 'use_gcrot'     : True,
    #                 'out_file'      : 'kona_normal_krylov.dat',
    #                 'subspace_size' : 20,
    #                 'max_outer'     : 20,
    #                 'max_recycle'   : 20,
    #                 'max_matvec'    : 100,
    #                 'check_res'     : True,
    #                 'rel_tol'       : 1e-8,
    #                 'abs_tol'       : 1e-12,
    #             },
    #             'tangent-step' : {
    #                 'out_file'      : 'kona_tangent_krylov.dat',
    #                 'subspace_size' : 20,
    #                 'check_res'     : True,
    #                 'rel_tol'       : 1e-8,
    #                 'abs_tol'       : 1e-12,
    #             }
    #         },
    #
    #         'verify' : {
    #             'primal_vec'    : True,
    #             'state_vec'     : True,
    #             'dual_vec'      : True,
    #             'gradients'     : True,
    #             'pde_jac'       : True,
    #             'cnstr_jac'     : True,
    #             'red_grad'      : True,
    #             'lin_solve'     : True,
    #             'out_file'      : sys.stdout,
    #         },
    #     }
    #
    #     algorithm = CompositeStepRSNK
    #     # algorithm = Verifier
    #     optimizer = Optimizer(solver, algorithm, optns)
    #     optimizer.solve()
    #
    #     print solver.curr_design
    #
    #     expected = numpy.array([1.977, 0., 0.])
    #     diff = abs(solver.curr_design - expected)
    #     self.assertTrue(max(diff) < 5e-2)
    #
    # def test_exponential_constrained(self):
    #
    #     feasible = True
    #     if feasible:
    #         init_x = [1., 1.]
    #     else:
    #         init_x = [-1., -1.]
    #
    #     solver = ExponentialConstrained(init_x=init_x)
    #
    #     optns = {
    #         'info_file' : 'kona_info.dat',
    #         'max_iter' : 50,
    #         'opt_tol' : 1e-5,
    #         'feas_tol' : 1e-5,
    #
    #         'globalization' : 'linesearch',
    #
    #         'trust' : {
    #             'init_radius' : 1.0,
    #             'max_radius' : 10.0,
    #             'min_radius' : 1e-4,
    #         },
    #
    #         'penalty' : {
    #             'mu_init' : 1.0,
    #             'mu_pow' : 1e-8,
    #         },
    #
    #         'composite-step' : {
    #             'normal-step' : {
    #                 'precond'       : None,
    #                 'lanczos_size'  : 1,
    #                 'use_gcrot'   : True,
    #                 'out_file'      : 'kona_normal_krylov.dat',
    #                 'subspace_size' : 10,
    #                 'max_outer'     : 10,
    #                 'max_recycle'   : 10,
    #                 'max_matvec'    : 100,
    #                 'check_res'     : True,
    #                 'rel_tol'       : 1e-5,
    #                 'abs_tol'       : 1e-8,
    #             },
    #             'tangent-step' : {
    #                 'out_file'      : 'kona_tangent_krylov.dat',
    #                 'subspace_size' : 50,
    #                 'check_res'     : True,
    #                 'rel_tol'       : 1e-5,
    #                 'abs_tol'       : 1e-8,
    #             }
    #         },
    #     }
    #
    #     algorithm = CompositeStepRSNK
    #     optimizer = Optimizer(solver, algorithm, optns)
    #     optimizer.solve()
    #
    #     print solver.curr_design
    #
    #     expected = numpy.zeros(solver.num_primal)
    #     diff = abs(solver.curr_design - expected)
    #     self.assertTrue(max(diff) < 1e-4)
    #
    # def test_with_simple_constrained(self):
    #
    #     feasible = True
    #     if feasible:
    #         init_x = [0.51, 0.52, 0.53]
    #     else:
    #         init_x = [1.51, 1.52, 1.53]
    #
    #     solver = SphereConstrained(init_x=init_x, ineq=True)
    #
    #     optns = {
    #         'info_file' : 'kona_info.dat',
    #         'max_iter' : 50,
    #         'opt_tol' : 1e-5,
    #         'feas_tol' : 1e-5,
    #
    #         'globalization' : 'linesearch',
    #
    #         'trust' : {
    #             'init_radius' : 4.0,
    #             'max_radius' : 1.0,
    #             'min_radius' : 1e-4,
    #         },
    #
    #         'penalty' : {
    #             'mu_init' : 1.0,
    #             'mu_pow' : 1e-8,
    #         },
    #
    #         'composite-step' : {
    #             'normal-step' : {
    #                 'precond'     : None,
    #                 'lanczos_size': 1,
    #                 'use_gcrot'   : True,
    #                 'out_file'    : 'kona_normal_krylov.dat',
    #                 'subspace_size'   : 10,
    #                 'max_outer'   : 10,
    #                 'max_recycle' : 10,
    #                 'max_matvec'  : 100,
    #                 'check_res'   : True,
    #                 'rel_tol'     : 1e-5,
    #                 'abs_tol'     : 1e-8,
    #             },
    #             'tangent-step' : {
    #                 'out_file'    : 'kona_tangent_krylov.dat',
    #                 'subspace_size'    : 50,
    #                 'check_res'   : True,
    #                 'rel_tol'     : 1e-5,
    #                 'abs_tol'     : 1e-8,
    #             }
    #         },
    #     }
    #
    #     algorithm = CompositeStepRSNK
    #     # algorithm = Verifier
    #     optimizer = Optimizer(solver, algorithm, optns)
    #     optimizer.solve()
    #
    #     print solver.curr_design
    #
    #     expected = -1.*numpy.ones(solver.num_primal)
    #     diff = abs(solver.curr_design - expected)
    #     self.assertTrue(max(diff) < 1e-3)

if __name__ == "__main__":
    unittest.main()
