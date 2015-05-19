import unittest
import numpy

from kona.examples import Simple2x2
from kona.linalg.memory import KonaMemory
from kona.algorithms import ReducedSpaceNewtonKrylov

class ReducedSpaceNewtonKrylovTestCase(unittest.TestCase):

    def test_RSNK_with_Simple2x2(self):

        solver = Simple2x2()
        km = KonaMemory(solver)

        optns = {
            'max_iter' : 50,
            'primal_tol' : 1e-12,
        }
        rsnk = ReducedSpaceNewtonKrylov(km.primal_factory, km.state_factory, optns)
        km.allocate_memory()
        rsnk.solve()

        self.failUnless('Untested')

if __name__ == "__main__":
    unittest.main()
