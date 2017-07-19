import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from kona.user import UserSolverIDF
from dummy_solver import DummySolver

class DesignVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 10, 0)
        self.km = km = KonaMemory(solver)

        km.design_lb = 0.

        km.primal_factory.request_num_vectors(3)
        km.state_factory.request_num_vectors(3)
        km.eq_factory.request_num_vectors(1)
        km.allocate_memory()

        # can't create bear KonaVectors because the memory manager doesn't
        # like them, so I'll just use the DesignVector to test the
        # KonaVectorMethods
        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()

    def test_check_type(self):
        '''DesignVector type checking'''
        try:
            self.pv._check_type(self.sv)
        except TypeError as err:
            self.assertEqual(
                str(err),
                "Vector type mismatch. " +
                "Must be <class 'kona.linalg.vectors.common.DesignVector'>")
        else:
            self.fail('TypeError expected')

        try:
            self.pv._check_type(self.pv)
        except TypeError:
            self.fail('Type error not expected')

    # NOTE: If test_inner is not working, none of the other tests will!!!
    def test_inner(self):
        '''DesignVector inner product'''
        # have to manually poke the data here
        # so the test doesn't rely on any other methods
        self.pv.base.data = 2*np.ones(10)
        self.assertEqual(self.pv.inner(self.pv), 40)

        # have to manually poke the data here
        # so the test doesn't rely on any other methods
        self.pv.base.data = -2*np.ones(10)
        self.assertEqual(self.pv.inner(self.pv), 40)

    def test_norm2(self):
        '''DesignVector L2 norm'''
        self.pv.equals(3)
        self.assertEqual(self.pv.norm2, np.linalg.norm(3*np.ones(10)))

    def test_equals(self):
        '''DesignVector scalar-value assignment'''
        self.pv.equals(3)
        self.assertEqual(self.pv.inner(self.pv), 90)

    def test_plus(self):
        '''DesignVector elementwise summation'''
        self.pv.equals(1)
        self.pv.plus(self.pv)
        self.assertEqual(self.pv.inner(self.pv), 40)

    def test_minus(self):
        '''DesignVector elementwise subtraction'''
        self.pv.equals(1)
        self.pv.minus(self.pv) # special case
        self.assertEqual(self.pv.inner(self.pv), 0)

        self.pv.equals(1)
        pv2 = self.km.primal_factory.generate()
        pv2.equals(5)
        pv2.minus(self.pv)
        self.assertEqual(pv2.inner(self.pv), 40)

    def test_times(self):
        '''DesignVector scalar and elementwise multiplication'''
        self.pv.equals(2)
        self.pv.times(5)
        self.assertEqual(self.pv.inner(self.pv), 1000)

        self.pv.equals(2.0)
        self.pv.times(5.0)
        self.assertEqual(self.pv.inner(self.pv), 1000.0)

        self.pv.equals(2.0)
        self.pv.times(self.pv)
        self.assertEqual(self.pv.inner(self.pv), 160.0)

    def test_divide_by(self):
        '''DesignVector scalar division'''
        self.pv.equals(10)
        self.pv.divide_by(5)
        self.assertEqual(self.pv.inner(self.pv), 40)

    def test_exp(self):
        '''DesignVector elementwise exponential'''
        self.pv.equals(0.)
        self.pv.exp(self.pv)
        self.assertEqual(self.pv.inner(self.pv), 10.0)

    def test_log(self):
        '''DesignVector elementwise natural logarithm'''
        self.pv.equals(1.)
        self.pv.log(self.pv)
        self.assertEqual(self.pv.inner(self.pv), 0.0)

    def test_equals_ax_p_by(self):
        '''DesignVector linear combination (ax + by)'''
        self.pv.equals(1)
        pv2 = self.km.primal_factory.generate()
        pv2.equals(1)

        pv3 = self.km.primal_factory.generate()
        pv3.equals(2)

        pv3.equals_ax_p_by(2, self.pv, 3, pv3)
        self.assertEqual(pv3.inner(self.pv), 80.0)

        pv2.equals_ax_p_by(2, self.pv, 3, pv2)
        self.assertEqual(pv2.inner(self.pv), 50.0)

    def test_init_design(self):
        '''DesignVector initial design assignment'''
        self.pv.equals_init_design()
        self.assertEqual(self.pv.inner(self.pv), 1000.0)

    def test_equals_objective_partial(self):
        '''DesignVector objective partial derivative'''
        at_design = self.km.primal_factory.generate()
        at_design.equals(1)
        at_state = self.sv
        at_state.equals(2)
        self.pv.equals_objective_partial(at_design, at_state)
        self.assertEqual(self.pv.inner(self.pv), 10.0)

    def test_equals_total_gradient(self):
        '''DesignVector objective total gradient'''
        at_design = self.km.primal_factory.generate()
        at_design.equals(1)
        at_state = self.km.state_factory.generate()
        at_state.equals(2)
        primal_work = self.km.primal_factory.generate()
        at_adjoint = self.km.state_factory.generate()
        at_adjoint.equals(3)
        self.pv.equals_total_gradient(at_design, at_state, at_adjoint)
        self.assertEqual(self.pv.inner(self.pv), 160.0)

    def test_equals_lagrangian_total_gradient(self):
        '''DesignVector lagrangian total gradient'''
        at_design = self.km.primal_factory.generate()
        at_design.equals(1)
        at_state = self.km.state_factory.generate()
        at_state.equals(2)
        primal_work = self.km.primal_factory.generate()
        at_adjoint = self.km.state_factory.generate()
        at_adjoint.equals(3)
        at_dual = self.km.eq_factory.generate()
        at_dual.equals(4)
        self.pv.equals_lagrangian_total_gradient(at_design, at_state, at_dual,
                                                 at_adjoint)
        self.assertEqual(self.pv.inner(self.pv), 19360.0)

    def test_enforce_bounds(self):
        '''DesignVector bound enforcement'''
        self.pv.equals(-1.)
        self.pv.enforce_bounds()
        zero = self.pv.norm2
        self.assertEqual(zero, 0.0)

class TestCaseDesignVectorIDF(unittest.TestCase):

    def setUp(self):
        solver = UserSolverIDF(5, 10, 10)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.eq_factory.request_num_vectors(1)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.dv = km.eq_factory.generate()

    def test_restrict_to_design(self):
        '''DesignVector IDF restriction (1/2)'''
        self.pv.equals(5)
        self.pv.restrict_to_design()
        inner_prod = self.pv.inner(self.pv)
        expected_prod = 5.*5.*5
        self.assertEqual(inner_prod, expected_prod)

    def test_restrict_to_target(self):
        '''DesignVector IDF restriction (2/2)'''
        self.pv.equals(5)
        self.pv.restrict_to_target()
        inner_prod = self.pv.inner(self.pv)
        expected_prod = 5.*5.*10
        self.assertEqual(inner_prod, expected_prod)

    def test_convert_to_dual(self):
        '''DesignVector IDF conversion'''
        self.pv.equals(2)
        self.dv.equals(1)
        self.pv.convert_to_dual(self.dv)
        inner_prod = self.dv.inner(self.dv)
        expected_prod = 40.
        self.assertEqual(inner_prod, expected_prod)

if __name__ == "__main__":
    unittest.main()