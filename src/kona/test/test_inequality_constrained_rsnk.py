import numpy
import unittest

from kona.algorithms import ConstrainedRSNK
from kona.examples import SimpleConstrained, ExponentialConstrained
from kona import Optimizer
from kona.algorithms.util.merit import AugmentedLagrangian

class InequalityConstrainedRSNKTestCase(unittest.TestCase):

    def test_placeholder(self):
        self.failUnless('Untested')

    # def test_with_exponential_constrained(self):
    #
    #     feasible = False
    #     if feasible:
    #         init_x = [1., 1.]
    #     else:
    #         init_x = [-1., -1.]
    #
    #     solver = ExponentialConstrained(init_x=init_x)
    #
    #     optns = {
    #         'info_file' : 'kona_info.dat',
    #         'max_iter' : 30,
    #         'primal_tol' : 1e-5,
    #         'constraint_tol' : 1e-5,
    #
    #         'trust' : {
    #             'init_radius' : 2.0,
    #             'max_radius' : 10.0,
    #             'min_radius' : 1e-3,
    #         },
    #
    #         'merit_function' : {
    #             'type' : AugmentedLagrangian
    #         },
    #
    #         'aug_lag' : {
    #             'mu_init' : 100.0,
    #             'mu_pow' : 1.0,
    #             'mu_max' : 1e5,
    #         },
    #
    #         'reduced' : {
    #             'precond'       : None,
    #             'product_fac'   : 0.001,
    #             'lambda'        : 0.0,
    #             'scale'         : 0.0,
    #             'nu'            : 0.95,
    #             'dynamic_tol'   : False,
    #         },
    #
    #         'krylov' : {
    #             'out_file'      : 'kona_krylov.dat',
    #             'max_iter'      : 10,
    #             'rel_tol'       : 0.0095,
    #             'check_res'     : True,
    #         },
    #     }
    #
    #     algorithm = ConstrainedRSNK
    #     optimizer = Optimizer(solver, algorithm, optns)
    #     optimizer.solve()
    #
    #     print solver.curr_design
    #
    #     expected = numpy.zeros(solver.num_primal)
    #     diff = abs(solver.curr_design - expected)
    #     self.assertTrue(max(diff) < 1e-4)

    # def test_with_simple_constrained(self):
    #
    #     feasible = True
    #     if feasible:
    #         init_x = [0.51, 0.52, 0.53]
    #     else:
    #         init_x = [1.51, 1.52, 1.53]
    #
    #     solver = SimpleConstrained(init_x=init_x, ineq=True)
    #
    #     optns = {
    #         'info_file' : 'kona_info.dat',
    #         'max_iter' : 30,
    #         'primal_tol' : 1e-6,
    #         'constraint_tol' : 1e-6,
    #
    #         'trust' : {
    #             'init_radius' : 1.0,
    #             'max_radius' : 4.0,
    #             'min_radius' : 1e-5,
    #         },
    #
    #         'aug_lag' : {
    #             'mu_init' : 10.0,
    #             'mu_pow' : 0.5,
    #             'mu_max' : 1e5,
    #         },
    #
    #         'reduced' : {
    #             'precond'       : None,
    #             'product_fac'   : 0.001,
    #             'lambda'        : 0.0,
    #             'scale'         : 0.0,
    #             'nu'            : 0.95,
    #             'dynamic_tol'   : False,
    #         },
    #
    #         'krylov' : {
    #             'out_file'      : 'kona_krylov.dat',
    #             'max_iter'      : 10,
    #             'rel_tol'       : 0.0095,
    #             'check_res'     : True,
    #         },
    #     }
    #
    #     algorithm = ConstrainedRSNK
    #     optimizer = Optimizer(solver, algorithm, optns)
    #     optimizer.solve()
    #
    #     print solver.curr_design
    #
    #     expected = -1.*numpy.ones(solver.num_primal)
    #     diff = abs(solver.curr_design - expected)
    #     self.assertTrue(max(diff) < 1e-5)

if __name__ == "__main__":
    unittest.main()
