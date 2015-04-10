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

        pf.request_num_vectors(3)
        sf.request_num_vectors(1)
        om = ObjectiveMerit(pf, sf)

        km.allocate_memory()

        p = pf.generate()
        p.equals(1)

        x_start = pf.generate()
        x_start.equals(2)

        #make up a a direction

        om.reset(p, x_start)

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
