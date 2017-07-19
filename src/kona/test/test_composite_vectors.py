import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from dummy_solver import DummySolver
from kona.linalg.vectors.composite import CompositeVector

class CompositeVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 5, 0)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(3)
        km.eq_factory.request_num_vectors(3)
        km.allocate_memory()

        self.pv1 = km.primal_factory.generate()
        self.pv1.base.data[:] = 2*np.ones(10)
        self.eq1 = km.eq_factory.generate()
        self.eq1.base.data[:] = 3*np.ones(5)
        self.comp_vec1 = CompositeVector([self.pv1, self.eq1])

        self.pv2 = km.primal_factory.generate()
        self.pv2.base.data[:] = 2*np.ones(10)
        self.eq2 = km.eq_factory.generate()
        self.eq2.base.data[:] = 2*np.ones(5)
        self.comp_vec2 = CompositeVector([self.pv2, self.eq2])

        self.diff_comp1 = CompositeVector([self.pv1, self.pv2])
        self.diff_comp2 = CompositeVector([self.eq1, self.eq2])

    def test_check_type(self):
        '''CompositeVector type checking'''
        try:
            self.comp_vec2._check_type(self.pv1)
        except TypeError as err:
            self.assertEqual(
                str(err),
                "CompositeVector() >> Wrong vector type. Must be " +
                "<class 'kona.linalg.vectors.composite.CompositeVector'>")
        else:
            self.fail('TypeError expected')

        try:
            self.diff_comp1._check_type(self.diff_comp2)
        except TypeError as err:
            self.assertEqual(
                str(err),
                "CompositeVector() >> " +
                "Mismatched internal vectors!")
        else:
            self.fail('TypeError expected')

    def test_equals(self):
        '''CompositeVector vector-value assignment'''
        self.comp_vec2.equals(self.comp_vec1)

        err = self.eq2.base.data - self.eq1.base.data
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.pv2.base.data - self.pv1.base.data
        self.assertEqual(np.linalg.norm(err), 0)

    def test_plus(self):
        '''CompositeVector elementwise summation'''
        self.comp_vec2.plus(self.comp_vec1)

        err = self.pv2.base.data - 4*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq2.base.data - 5*np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

    def test_minus(self):
        '''CompositeVector elementwise subtraction'''
        self.comp_vec2.minus(self.comp_vec1)

        err = self.pv2.base.data - 0*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq2.base.data - -1*np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

    def test_times_vector(self):
        '''CompositeVector elementwise multiplication'''
        self.comp_vec2.times(self.comp_vec1)
        self.assertEqual(self.comp_vec2.inner(self.comp_vec2), 340.)

    def test_times_scalar(self):
        '''CompositeVector scalar multiplication'''
        self.comp_vec2.times(3)
        err = self.pv2.base.data - 6*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq2.base.data - 6*np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

        self.comp_vec1.times(3.0)
        err = self.pv1.base.data - 6*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq1.base.data - 9*np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

    def test_divide_by(self):
        '''CompositeVector elementwise division'''
        self.comp_vec2.divide_by(2)
        err = self.pv2.base.data - 1*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq2.base.data - 1*np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

    def test_equals_ax_p_by(self):
        '''CompositeVector linear combination (ax + by)'''
        self.comp_vec2.equals_ax_p_by(2, self.comp_vec1, 2, self.comp_vec2)

        err = self.pv2.base.data - 8*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq2.base.data - 10*np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

    def test_inner(self):
        '''CompositeVector inner product'''
        ip = self.comp_vec2.inner(self.comp_vec1)
        self.assertEqual(ip, 70)

    def test_norm2(self):
        '''CompositeVector L2 norm'''
        ip = self.comp_vec2.norm2
        self.assertEqual(ip, 60**.5)



if __name__ == "__main__":
    unittest.main()
