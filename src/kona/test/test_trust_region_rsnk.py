import unittest
import numpy

from kona.linalg.memory import KonaMemory
from kona.algorithms import ReducedSpaceNewtonKrylov
from kona.examples import Rosenbrock, Spiral

class ReducedSpaceNewtonKrylovTestCase(unittest.TestCase):

    def test_RSNK_with_Rosenbrock(self):

        num_design = 2
        solver = Rosenbrock(num_design)
        km = KonaMemory(solver)

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
        rsnk = ReducedSpaceNewtonKrylov(
            km.primal_factory, km.state_factory, None, optns)
        km.allocate_memory()
        rsnk.solve()

        diff = abs(solver.curr_design - numpy.ones(num_design))
        self.assertTrue(max(diff) < 1e-5)

    def test_RSNK_with_Spiral(self):

        solver = Spiral()
        km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(4)
        km.state_factory.request_num_vectors(4)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 50,
            'opt_tol' : 1e-8,

            'trust' : {
                'init_radius' : 1.0,
                'max_radius' : 4.0,
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

        rsnk = ReducedSpaceNewtonKrylov(
            km.primal_factory, km.state_factory, None, optns)
        km.allocate_memory()
        rsnk.solve()

        diff = abs(solver.curr_design)
        self.assertTrue(max(diff) < 1e-5)

if __name__ == "__main__":
    unittest.main()
