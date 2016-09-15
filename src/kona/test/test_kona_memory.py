import gc
import unittest

from kona.linalg.memory import KonaMemory
from kona.linalg.vectors.common import DesignVector
from kona.user.user_solver import UserSolver

class VectorFactoryTestCase(unittest.TestCase):

    def test_generate(self):
        ndv = 2
        solver = UserSolver(ndv)
        km = KonaMemory(solver)
        vf = km.primal_factory

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

    def test_error_generate(self):
        ndv = 2
        solver = UserSolver(ndv)
        km = KonaMemory(solver)
        vf = km.primal_factory

        vf.request_num_vectors(1)

        try:
            vf.generate()
        except RuntimeError as err:
            self.assertEqual(
                str(err),
                'VectorFactory() >> ' +
                'Must allocate memory before generating vector.')
        else:
            self.fail('RuntimeError')

        km.allocate_memory()

        try:
            km.allocate_memory()
        except RuntimeError as err:
            self.assertEqual(
                str(err),
                'Memory already allocated, can-not re-allocate')
        else:
            self.fail("RuntimeError expected")

    def test_too_many_vectors(self):
        ndv = 2
        solver = UserSolver(ndv)
        km = KonaMemory(solver)
        vf = km.primal_factory

        vf.request_num_vectors(2)
        km.allocate_memory()

        try:
            vec0 = vf.generate()
            vec1 = vf.generate()
            vec2 = vf.generate()
        except MemoryError as err:
            self.assertEqual(
                str(err),
                'No more vector memory available. ' +
                'Allocate more vectors in your algorithm initialization')
        else:
            self.fail('MemoryError expected')


if __name__ == "__main__":
    unittest.main()
