import gc
import unittest

import numpy as np

from kona.linalg.memory import KonaMemory
from kona.linalg.vectors.common import DesignVector
from kona.user.user_solver import UserSolver

class VectorFactoryTestCase(unittest.TestCase):

    def test_generate(self):
        solver = UserSolver()
        km = KonaMemory(solver)
        vf = km.design_factory

        vf.request_num_vectors(10)
        vf.request_num_vectors(2)
        km.allocate_memory()
        self.assertEqual(len(km.vector_stack[DesignVector]), 12)

        vec0 = vf.generate()
        vec1 = vf.generate()
        vec2 = vf.generate()

        self.assertEqual(len(km.vector_stack[DesignVector]), 9)

        del vec0, vec1, vec2
        # have to make sure things get cleaned up here to avoid race conditions
        gc.collect()
        self.assertEqual(len(km.vector_stack[DesignVector]), 12)

# class KonaMemoryTestCase(self):
#
#     def test_kona


if __name__ == "__main__":
    unittest.main()
