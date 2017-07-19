import unittest
import numpy as np

from kona.user import BaseVector

class AbsVectorTestCase(unittest.TestCase):
    '''Test case that can be used for any base abstract vector class'''

    def setUp(self):

        self.x_vec = BaseVector(val=1, size=10) # initial value for the vector
        self.y_vec = BaseVector(val=2, size=10)
        self.z_vec = BaseVector(val=np.linspace(0,10,10), size=10)

    def tearDown(self):
        pass

    def test_inner_product(self):
        '''BaseVector inner product'''
        val = self.x_vec.inner(self.x_vec)
        self.assertEqual(val, 10)

        val = self.x_vec.inner(self.y_vec)
        self.assertEqual(val, 20)

    def test_times_scalar_equals(self):
        '''BaseVector scalar multiplication'''
        self.x_vec.times_scalar(3)
        self.assertEqual(self.x_vec.inner(self.x_vec), 90)

    def test_times_vector_equals(self):
        '''BaseVector elementwise vector multiplication'''
        self.x_vec.times_vector(self.y_vec)
        self.assertEqual(self.x_vec.inner(self.x_vec), 40)

    def test_plus_equals(self):
        '''BaseVector elementwise summation'''
        self.x_vec.plus(self.y_vec)
        self.assertEqual(self.x_vec.inner(self.x_vec), 90)

    def test_assignment(self):
        '''BaseVector value assignment'''
        self.z_vec.equals_value(15)
        self.assertEqual(self.z_vec.inner(self.z_vec), 2250)

        self.z_vec.equals_vector(self.x_vec)
        self.z_vec.times_scalar(2)
        self.assertEqual(self.z_vec.inner(self.z_vec), 40)

        self.z_vec.equals_ax_p_by(2, self.x_vec, 3, self.y_vec)

    def test_bad_value(self):
        '''BaseVector invalid value error message'''
        try:
            BaseVector(size=10, val="s")
        except ValueError as err:
            self.assertEqual(
                str(err),
                'val must be a scalar or array like, but was given as type str')

        try:
            BaseVector(size=10, val=np.ones(12))
        except ValueError as err:
            self.assertEqual(
                str(err),
                'size given as 10, but length of value 12')

if __name__ == "__main__":
    unittest.main()
