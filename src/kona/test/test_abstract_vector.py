import unittest

import numpy as np

from kona.user import BaseVector
from kona.user import BaseAllocator
# from kona.user_vectors.petsc_vector import NumpyVector # should follow this exact interface

class AbsVectorTestCase(unittest.TestCase):
    '''Test case that can be used for any base abstract vector class'''

    def setUp(self):

        self.x_vec = BaseVector(val=1, size=10) #initial value for the vector
        self.y_vec = BaseVector(val=2, size=10)
        self.z_vec = BaseVector(val=np.linspace(0,10,10), size=10)

    def tearDown(self):
        pass

    def test_inner_product(self):
        val = self.x_vec.inner(self.x_vec)
        self.assertEquals(val, 10)

        val = self.x_vec.inner(self.y_vec)
        self.assertEquals(val, 20)

    def test_times_equals(self):
        self.x_vec.times(3)
        norm = self.x_vec.inner(self.x_vec)
        self.assertEquals(self.x_vec.inner(self.x_vec), 90)

    def test_plus_equals(self):
        self.x_vec.plus(self.y_vec)
        self.assertEquals(self.x_vec.inner(self.x_vec), 90)

    def test_assignment(self):

        x_vec = self.x_vec

        self.z_vec.equals_value(15)
        self.assertEquals(self.z_vec.inner(self.z_vec), 2250)

        self.z_vec.equals_vector(self.x_vec)
        self.z_vec.times(2)
        self.assertEquals(self.z_vec.inner(self.z_vec), 40)

        self.z_vec.equals_ax_p_by(2, self.x_vec, 3, self.y_vec)

    def test_bad_value(self):
        try:
            BaseVector(size=10, val="s")
        except ValueError as err:
            self.assertEqual(str(err), 'val must be a scalar or array like, but was given as type str')

        try:
            BaseVector(size=10, val=np.ones(12))
        except ValueError as err:
            self.assertEqual(str(err), 'size given as 10, but length of value 12')


class TestCaseProblemAllocator(unittest.TestCase):

    def setUp(self):
        self.alloc = BaseAllocator(3, 4, 5)

    def test_primal_vec(self):
        base_var = self.alloc.alloc_primal(1)[0]
        self.assertTrue(isinstance(base_var, BaseVector))

        self.assertEqual(base_var.data.shape[0], 3)

    def test_state_vec(self):
        base_var = self.alloc.alloc_state(1)[0]
        self.assertTrue(isinstance(base_var, BaseVector))

        self.assertEqual(base_var.data.shape[0], 4)

    def test_dual_vec(self):
        base_var = self.alloc.alloc_dual(1)[0]
        self.assertTrue(isinstance(base_var, BaseVector))

        self.assertEqual(base_var.data.shape[0], 5)

if __name__ == "__main__":
    unittest.main()
