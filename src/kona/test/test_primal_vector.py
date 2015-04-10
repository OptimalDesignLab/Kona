import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from kona.user import UserSolverIDF
from dummy_solver import DummySolver

class PrimalVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 10, 0)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(3)
        km.state_factory.request_num_vectors(1)
        km.allocate_memory()

        #can't create bear KonaVectors because the memory manager doesn't
        # like them, so I'll just use the PrimalVector to test the
        # KonaVectorMethods
        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()

    def test_check_type(self):
        try:
            self.pv._check_type(self.sv)
        except TypeError as err:
            self.assertEqual(str(err), "Vector type mismatch. Must be <class 'kona.linalg.vectors.common.PrimalVector'>")
        else:
            self.fail('TypeError expected')

        try:
            self.pv._check_type(self.pv)
        except TypeError:
            self.fail('Type error not expected')

    # NOTE: If test_inner is not working, none of the other tests will!!!
    def test_inner(self):
        self.pv._data.data = 2*np.ones(10) # have to manually poke the data here, so the test doesn't rely on any other methods
        self.assertEqual(self.pv.inner(self.pv), 40)

        self.pv._data.data = -2*np.ones(10) # have to manually poke the data here, so the test doesn't rely on any other methods
        self.assertEqual(self.pv.inner(self.pv), 40)

    def test_norm2(self):
        self.pv.equals(3)
        self.assertEqual(self.pv.norm2, np.linalg.norm(3*np.ones(10)))

    def test_equals(self):
        self.pv.equals(3)
        self.assertEqual(self.pv.inner(self.pv), 90)

    def test_plus(self):
        self.pv.equals(1)
        self.pv.plus(self.pv)
        self.assertEqual(self.pv.inner(self.pv), 40)

    def test_minus(self):
        self.pv.equals(1)
        self.pv.minus(self.pv) #special case
        self.assertEqual(self.pv.inner(self.pv), 0)

        self.pv.equals(1)
        pv2 = self.km.primal_factory.generate()
        pv2.equals(5)
        pv2.minus(self.pv)
        self.assertEqual(pv2.inner(self.pv), 40)


    def test_times(self):
        self.pv.equals(2)
        self.pv.times(5)
        self.assertEqual(self.pv.inner(self.pv), 1000)

        self.pv.equals(2.0)
        self.pv.times(5.0)
        self.assertEqual(self.pv.inner(self.pv), 1000.0)

        try:
            self.pv.times(self.pv)
        except TypeError as err:
            self.assertEqual(str(err), 'Argument must be a float.')
        else:
            self.fail('TypeError expected')

    def test_divide_by(self):
        self.pv.equals(10)
        self.pv.divide_by(5)
        self.assertEqual(self.pv.inner(self.pv), 40)

    def test_equals_ax_p_by(self):
        self.pv.equals(1)
        pv2 = self.km.primal_factory.generate()
        pv2.equals(1)

        pv3 = self.km.primal_factory.generate()
        pv3.equals(2)

        pv3.equals_ax_p_by(2, self.pv, 3, pv3)
        self.assertEqual(pv3.inner(self.pv), 80)

        pv2.equals_ax_p_by(2, self.pv, 3, pv2)
        self.assertEqual(pv2.inner(self.pv), 50)

    def test_init_design(self):
        self.pv.equals_init_design()
        self.assertEqual(self.pv.inner(self.pv), 160)

    def test_equals_objective_gradient(self):
        at_design = self.km.primal_factory.generate()
        at_design.equals(1)
        at_state = self.sv
        at_state.equals(2)
        self.pv.equals_objective_partial(at_design, at_state)
        self.assertEqual(self.pv.inner(self.pv), 4000)

    def test_equals_reduced_gradient(self):
        self.fail('Untested')

    def test_equals_lagrangian_reduced_gradient(self):
        self.fail('Untested')

class TestCasePrimalVectorIDF(unittest.TestCase):

    def setUp(self):
        solver = UserSolverIDF(5, 10, 0)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.dual_factory.request_num_vectors(1)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.dv = km.dual_factory.generate()

    def test_restrict_target_state(self):
        self.pv.equals(5)
        self.pv.restrict_target_state()
        inner_prod = self.pv.inner(self.pv)
        expected_prod = 5.*5.*5
        self.assertEqual(inner_prod, expected_prod)

    def test_restrict_real_design(self):
        self.pv.equals(5)
        self.pv.restrict_real_design()
        inner_prod = self.pv.inner(self.pv)
        expected_prod = 5.*5.*10
        self.assertEqual(inner_prod, expected_prod)

    def test_convert(self):
        self.pv.equals(1)
        self.dv.equals(2)
        self.pv.convert(self.dv)
        inner_prod = self.pv.inner(self.pv)
        expected_prod = (1.*1.*5) + (2.*2.*10)
        self.assertEqual(inner_prod, expected_prod)

if __name__ == "__main__":
    unittest.main()
