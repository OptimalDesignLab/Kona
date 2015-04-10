import unittest

from kona.linalg.memory import KonaMemory
from kona.linalg.vectors.common import objective_value

from kona.algorithms.util.merit import ObjectiveMerit
from kona.examples.simple_2by2 import Simple2x2


class ObjectiveMeritTestCase(unittest.TestCase):

    def test_eval_func(self):

        solver = Simple2x2()
        km = KonaMemory(solver)
        pf = km.primal_factory
        sf = km.state_factory

        pf.request_num_vectors(10)
        sf.request_num_vectors(10)
        om = ObjectiveMerit(pf, sf)

        km.allocate_memory()

        p = pf.generate()
        p.equals(1)

        x_start = pf.generate()
        x_start.equals(2)

        u_start = sf.generate()
        u_start.equals_primal_solution(x_start)

        grad = pf.generate()
        adjoint = sf.generate()
        state_work = sf.generate()
        adjoint.equals_adjoint_solution(x_start, u_start, state_work)

        primal_work = pf.generate()
        grad.equals_total_gradient(x_start, u_start, adjoint, primal_work)

        p_dot_grad = p.inner(grad)

        om.reset(p, x_start, u_start, p_dot_grad)

        merit = om.eval_func(1)

        p_expected = pf.generate()
        u_expected = sf.generate()
        p.equals(3) # x + alpha*p; 2 + 1*1
        u_expected.equals_primal_solution(p_expected)
        expected_value = objective_value(p_expected, u_expected)

        self.assertEqual(merit, expected_value)

    def test_eval_grad(self):
        self.failUnless('Untested')

if __name__ == "__main__":

    unittest.main()
