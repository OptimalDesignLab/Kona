import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from dummy_solver import DummySolver
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector

class CompositeDualVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 5, 5)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(1)
        km.state_factory.request_num_vectors(1)
        km.eq_factory.request_num_vectors(1)
        km.ineq_factory.request_num_vectors(2)
        km.allocate_memory()

        self.design = km.primal_factory.generate()
        self.slack = km.ineq_factory.generate()
        self.primal = CompositePrimalVector(self.design, self.slack)
        self.state = km.state_factory.generate()

        # set the evaluation point
        self.design.equals_init_design()
        self.state.equals_primal_solution(self.design)

        self.eq = km.eq_factory.generate()
        self.ineq = km.ineq_factory.generate()
        self.cdv = CompositeDualVector(self.eq, self.ineq)

    def test_bad_init_args(self):
        '''CompositeDualVector test for bad initial arguments'''
        try:
            CompositeDualVector(self.ineq, self.ineq)
        except TypeError as err:
            self.assertEqual(
                str(err),
                'CompositeDualVector() >> ' +
                'Unidentified equality constraint vector.')
        else:
            self.fail('TypeError expected')

        try:
            CompositeDualVector(self.eq, self.eq)
        except TypeError as err:
            self.assertEqual(
                str(err),
                'CompositeDualVector() >> ' +
                'Unidentified inequality constraint vector.')
        else:
            self.fail('TypeError expected')

    def test_equals_constraints_with_design(self):
        '''CompositeDualVector test for constraint evaluation (1/2)'''
        self.cdv.equals_constraints(self.design, self.state)
        exp_norm = np.sqrt(5. * 200.**2)
        self.assertEqual(self.cdv.eq.norm2, exp_norm)
        self.assertEqual(self.cdv.ineq.norm2, exp_norm)

    def test_equals_constraints_with_primal(self):
        '''CompositeDualVector test for constraint evaluation (2/2)'''
        self.cdv.equals_constraints(self.primal, self.state)
        exp_norm = np.sqrt(5. * 200.**2)
        self.assertEqual(self.cdv.eq.norm2, exp_norm)
        self.assertEqual(self.cdv.ineq.norm2, exp_norm)


if __name__ == "__main__":
    unittest.main()
