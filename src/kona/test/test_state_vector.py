import unittest

from kona.linalg.memory import KonaMemory

from dummy_solver import DummySolver

class StateVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 10, 0)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(3)
        km.allocate_memory()

        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()
        self.sv_work = km.state_factory.generate()

    def test_equals_objective_partial(self):
        at_design = self.pv
        at_design.equals(1)
        at_state = self.sv_work
        at_state.equals(2)
        self.sv.equals_objective_partial(at_design, at_state)
        self.assertEqual(self.sv.inner(self.sv), 10.0)

    def test_equals_residual(self):
        at_design = self.pv
        at_design.equals(1)
        at_state = self.sv_work
        at_state.equals(2)
        self.sv.equals_residual(at_design, at_state)
        self.assertEqual(self.sv.inner(self.sv), 90.0)

    def test_equals_primal_solution(self):
        at_design = self.pv
        at_design.equals(2)
        self.sv.equals_primal_solution(at_design)
        self.assertEqual(self.sv.inner(self.sv), 40.0)

    def test_equals_adjoint_solution(self):
        at_design = self.pv
        at_design.equals(1)
        at_state = self.sv_work
        at_state.equals(2)
        state_work = self.km.state_factory.generate()
        self.sv.equals_objective_adjoint(at_design, at_state, state_work)
        self.assertEqual(self.sv.inner(self.sv), 10.0)

if __name__ == "__main__":
    unittest.main()
