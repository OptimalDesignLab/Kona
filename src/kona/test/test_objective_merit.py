import unittest

from kona.linalg.memory import KonaMemory
from kona.linalg.common import objective_value

from kona.algorithms.util.merit import ObjectiveMerit
from kona.examples.simple_2by2 import Simple2x2


class ObjectiveMeritTestCase(unittest.TestCase):

    def setUp(self):
        solver = Simple2x2()
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.sf = self.km.state_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.merit = ObjectiveMerit(self.pf, self.sf)

        self.km.allocate_memory()

        p = self.pf.generate()
        p.equals(1)

        x_start = self.pf.generate()
        x_start.equals(2)

        u_start = self.sf.generate()
        u_start.equals_primal_solution(x_start)

        grad = self.pf.generate()
        adjoint = self.sf.generate()
        state_work = self.sf.generate()
        adjoint.equals_adjoint_solution(x_start, u_start, state_work)

        primal_work = self.pf.generate()
        grad.equals_total_gradient(x_start, u_start, adjoint, primal_work)
        self.p_dot_grad_init = p.inner(grad)
        self.merit_val_init = objective_value(x_start, u_start)

        self.merit.reset(p, x_start, u_start, self.p_dot_grad_init)

    def test_eval_func(self):
        # get merit value from merit object
        merit_value = self.merit.eval_func(1)
        # calculate expected merit value
        x_expected = self.pf.generate()
        u_expected = self.sf.generate()
        x_expected.equals(3) # x + alpha*p; 2 + 1*1
        u_expected.equals_primal_solution(x_expected)
        expected_value = objective_value(x_expected, u_expected)
        # compare the two
        self.assertEqual(merit_value, expected_value)

    def test_eval_grad(self):
        # get merit grad from merit object
        merit_value = self.merit.eval_grad(1)
        # calculate expected merit value
        x_expected = self.pf.generate()
        u_expected = self.sf.generate()
        x_expected.equals(3) # x + alpha*p; 2 + 1*1
        u_expected.equals_primal_solution(x_expected)
        adjoint = self.sf.generate()
        state_work = self.sf.generate()
        adjoint.equals_adjoint_solution(x_expected, u_expected, state_work)
        primal_work = self.pf.generate()
        grad = self.pf.generate()
        grad.equals_total_gradient(x_expected, u_expected, adjoint, primal_work)
        p_expected = self.pf.generate()
        p_expected.equals(1)
        expected_value = p_expected.inner(grad)
        # compare the two
        self.assertEqual(merit_value, expected_value)

    def test_unnecessary_func_eval(self):
        merit_value = self.merit.eval_func(1e-20)
        self.assertEqual(merit_value, self.merit_val_init)

    def test_unnecessary_grad_eval(self):
        merit_grad = self.merit.eval_grad(1e-20)
        self.assertEqual(merit_grad, self.p_dot_grad_init)

if __name__ == "__main__":

    unittest.main()
