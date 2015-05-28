import unittest
import numpy

from kona.linalg.memory import KonaMemory
from kona.algorithms import ReducedSpaceNewtonKrylov, ReducedSpaceQuasiNewton
from kona.examples import Rosenbrock, Spiral

class ReducedSpaceNewtonKrylovTestCase(unittest.TestCase):

    def test_RSNK_with_Rosenbrock(self):

        num_design = 2
        solver = Rosenbrock(num_design)
        km = KonaMemory(solver)

        optns = {
            'max_iter' : 50,
            'primal_tol' : 1e-12,

            'trust' : {
                'radius' : 1.0,
                'max_radius' : 2.0,
            },

            'krylov' : {
                'out_file'      : 'kona_krylov.dat',
                'max_iter'      : num_design,
                'rel_tol'       : 1e-2,
                'check_res'     : True,
                # STCG options
                'proj_cg'       : False,
            },
        }
        rsnk = ReducedSpaceNewtonKrylov(km.primal_factory, km.state_factory, optns)
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
            'max_iter' : 50,
            'primal_tol' : 1e-8,

            'trust' : {
                'radius' : 1.0,
                'max_radius' : 2.0,
            },

            'reduced' : {
                'precond' : None,
            },

            'krylov' : {
                'out_file'      : 'kona_krylov.dat',
                'max_iter'      : 10,
                'rel_tol'       : 0.5,
                'check_res'     : True,
                # STCG options
                'proj_cg'       : False,
            },
        }

        rsnk = ReducedSpaceNewtonKrylov(km.primal_factory, km.state_factory, optns)
        km.allocate_memory()
        rsnk.solve()

        diff = abs(solver.curr_design)
        self.assertTrue(max(diff) < 1e-5)

if __name__ == "__main__":
    unittest.main()
