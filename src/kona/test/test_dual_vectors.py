import unittest

from dummy_solver import DummySolver
from kona.linalg.memory import KonaMemory
from kona.user import UserSolverIDF

class DualVectorEQTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 10, 0)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(1)
        km.eq_factory.request_num_vectors(1)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()
        self.dv = km.eq_factory.generate()

    def test_equals_constraints(self):
        at_design = self.pv
        at_design.equals(1)
        at_state = self.sv
        at_state.equals(2)
        self.dv.equals_constraints(at_design, at_state)
        self.assertEqual(self.dv.inner(self.dv), 1000)

class DualVectorINEQTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 0, 10)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(1)
        km.ineq_factory.request_num_vectors(3)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()
        self.dv = km.ineq_factory.generate()
        self.mult = km.ineq_factory.generate()
        self.mang = km.ineq_factory.generate()

    def test_equals_constraints(self):
        at_design = self.pv
        at_design.equals(1)
        at_state = self.sv
        at_state.equals(2)
        self.dv.equals_constraints(at_design, at_state)
        self.assertEqual(self.dv.inner(self.dv), 1000)

    def test_equals_mangasarian(self):
        at_design = self.pv
        at_design.equals(1)
        at_state = self.sv
        at_state.equals(2)
        self.dv.equals_constraints(at_design, at_state)
        self.mult.equals(-9)
        self.mang.equals_mangasarian(self.dv, self.mult)
        for i in xrange(len(self.mang.base.data)):
            self.assertEqual(self.mang.base.data[i], -10) # linear Mangasarian
            # self.assertEqual(self.mang.base.data[i], -1730) # cubic Mangasarian
        self.mult.equals(-11)
        self.mang.equals_mangasarian(self.dv, self.mult)
        for i in xrange(len(self.mang.base.data)):
            self.assertEqual(self.mang.base.data[i], -11) # linear Mangasarian
            # self.assertEqual(self.mang.base.data[i], -2332) # cubic Mangasarian

class DualVectorEQTestCaseIDF(unittest.TestCase):

    def setUp(self):
        solver = UserSolverIDF(5, 10, 10)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(1)
        km.eq_factory.request_num_vectors(1)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()
        self.dv = km.eq_factory.generate()

    def test_convert_to_primal(self):
        self.pv.equals(5)
        self.dv.equals(1)
        self.dv.convert_to_design(self.pv)
        inner_prod = self.pv.inner(self.pv)
        expected_prod = 10.
        self.assertEqual(inner_prod, expected_prod)

if __name__ == "__main__":
    unittest.main()
