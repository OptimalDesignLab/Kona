import unittest
import numpy as np

from kona.linalg.memory import KonaMemory

from dummy_solver import DummySolver

class StateVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 10, 0)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(2)
        km.allocate_memory()

        #can't create bear KonaVectors because the memory manager doesn't
        # like them, so I'll just use the PrimalVector to test the
        # KonaVectorMethods
        self.pv = km.primal_factory.generate()
        self.sv = km.state_factory.generate()
        self.sv_work = km.state_factory.generate()

    def test_equals_objective_partial(self):
        self.fail('Untested')

    def test_equals_residual(self):
        self.fail('Untested')

    def test_equals_primal_solution(self):
        self.fail('Untested')

    def equals_adjoint_solution(self):
        self.fail('Untested')
