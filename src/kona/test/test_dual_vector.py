import unittest

from kona.linalg.memory import KonaMemory
from kona.user import UserSolverIDF

from dummy_solver import DummySolver

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
        self.assertEqual(self.dv.inner(self.dv), 9000)

class DualVectorINEQTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 0, 10)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(1)
        km.ineq_factory.request_num_vectors(1)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()
        self.dv = km.ineq_factory.generate()

    def test_equals_constraints(self):
        at_design = self.pv
        at_design.equals(1)
        at_state = self.sv
        at_state.equals(2)
        self.dv.equals_constraints(at_design, at_state)
        self.assertEqual(self.dv.inner(self.dv), 9000)

class DualVectorEQTestCaseIDF(unittest.TestCase):

    def setUp(self):
        solver = UserSolverIDF(5, 10, 0)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(1)
        km.eq_factory.request_num_vectors(1)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()
        self.dv = km.eq_factory.generate()

    def test_convert(self):
        self.pv.equals(5)
        self.dv.equals(1)
        self.dv.convert(self.pv)

if __name__ == "__main__":
    unittest.main()
