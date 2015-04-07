import unittest

import numpy as np

from kona.linalg.memory import VectorFactory, KonaMemory
from kona.linalg.vectors.common import DesignVector

class DummyUserSolver(object):

    def get_rank(self):
        return 0

class VectorFactoryTestCase(unittest.TestCase):

    def test_generate(self):
        solver = DummyUserSolver()
        km = KonaMemory(solver)
        vf = VectorFactor(km, vec_type=DesignVector)

        vf.request_num_vectors(10)

        vecs = vf.generate()
        self.assertEqual(len(vecs), 10)


# class KonaMemoryTestCase(self):
#
#     def test_kona


if __name__ == "__main__":
    unittest.main()
