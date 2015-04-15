import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from kona.user import UserSolverIDF
from dummy_solver import DummySolver
from kona.linalg.vectors.composite import ReducedKKTVector

class ReducedKKTVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 10, 10)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(3)
        km.dual_factory.request_num_vectors(3)
        km.allocate_memory()

        #can't create bear KonaVectors because the memory manager doesn't
        # like them, so I'll just use the PrimalVector to test the
        # KonaVectorMethods
        self.pv = km.primal_factory.generate()
        self.dv = km.dual_factory.generate()

        self.rkkt_vec = ReducedKKTVector(km, self.pv, self.dv)

    def test_check_type(self):

        try:
            self.rkkt_vec._check_type(self.pv)
        except TypeError as err:
            self.assertEquals(str(err), "ReducedKKTVector() >> Wrong vector type. Must be <class 'kona.linalg.vectors.composite.ReducedKKTVector'>")
        else:
            self.fail('TypeError expected')

    def test_equals(self):
        pass

    def test_plus(self):
        pass

    def test_minus(self):
        pass

    def test_times(self):
        pass

    def test_divide_by(self):
        pass

    def test_equals_ax_p_by(self):
        pass

    def test_inner(self):
        pass

    def test_norm2(self):
        pass

    def test_equals_initial_guess(self):
        pass


if __name__ == "__main__":
    unittest.main()
