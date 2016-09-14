import unittest

from kona.linalg.memory import KonaMemory
from kona.algorithms.util.linesearch import StrongWolfe
from kona.algorithms.util.merit import ObjectiveMerit

from kona.examples.simple_2by2 import Simple2x2

class StrongWolfeTestCase(unittest.TestCase):

    def setUp(self):
        solver = Simple2x2()
        km = KonaMemory(solver)
        self.pf = pf = km.primal_factory
        self.sf = sf = km.state_factory

        pf.request_num_vectors(10)
        sf.request_num_vectors(10)

        self.merit = ObjectiveMerit(pf, sf)

        km.allocate_memory()

        self.sw = StrongWolfe() # leave all settings with initial values

        search_dir = self.pf.generate()
        search_dir.base.data[:] = [-1,0]

        at_design = self.pf.generate()
        at_design.equals(1)

        at_state = self.sf.generate()
        at_state.equals_primal_solution(at_design)

        state_work = self.sf.generate()

        adjoint = self.sf.generate()
        adjoint.equals_adjoint_solution(at_design, at_state, state_work)

        primal_work = self.pf.generate()

        dfdx = self.pf.generate()
        dfdx.equals_total_gradient(at_design, at_state, adjoint, primal_work)
        self.sw.p_dot_dfdx = dfdx.inner(search_dir)

    def test_find_step_length(self):
        '''Check that it works when the search direction sign is flips'''

        search_dir = self.pf.generate()
        search_dir.base.data[:] = [4.25,0]

        at_design = self.pf.generate()
        at_design.equals(-2)

        at_state = self.sf.generate()
        at_state.equals_primal_solution(at_design)

        state_work = self.sf.generate()

        adjoint = self.sf.generate()
        adjoint.equals_adjoint_solution(at_design, at_state, state_work)

        primal_work = self.pf.generate()

        dfdx = self.pf.generate()
        dfdx.equals_total_gradient(at_design, at_state, adjoint, primal_work)
        self.sw.p_dot_dfdx = dfdx.inner(search_dir)

        self.merit.reset(search_dir, at_design, at_state, self.sw.p_dot_dfdx)

        alpha, n_iter = self.sw.find_step_length(self.merit)

        self.assertEqual(n_iter, 2)
        self.assertTrue(abs(alpha - 0.4) < 1.e-1)

    def test_bad_search_direction(self):

        search_dir = self.pf.generate()
        search_dir.base.data[:] = [-1,0]

        at_design = self.pf.generate()
        at_design.equals(1)

        at_state = self.sf.generate()
        at_state.equals_primal_solution(at_design)

        grad = self.pf.generate()
        at_adjoint = self.sf.generate()
        state_work = self.sf.generate()
        at_adjoint.equals_adjoint_solution(at_design, at_state, state_work)

        primal_work = self.pf.generate()
        grad.equals_total_gradient(at_design, at_state, at_adjoint, primal_work)

        p_dot_grad = search_dir.inner(grad)

        self.merit.reset(search_dir, at_design, at_state, p_dot_grad)

        self.merit.p_dot_grad *= -1
        try:
            alpha, n_iter = self.sw.find_step_length(self.merit)
        except ValueError as err:
            self.assertEqual(
                str(err),
                'search direction is not a descent direction')
        else:
            self.fail('ValueError expected')

    def test_no_merit_function(self):
        try:
            alpha, n_iter = self.sw.find_step_length(None)
        except ValueError as err:
            self.assertEqual(str(err), 'unknown merit_function type')
        else:
            self.fail('ValueError expected')

    def test_bad_alpha_init(self):
        self.sw.alpha_init = -2.
        try:
            alpha, n_iter = self.sw.find_step_length(None)
        except ValueError as err:
            self.assertEqual(
                str(err),
                'alpha_init must be greater than zero (0)')
        else:
            self.fail('ValueError expected')

    def test_bad_alpha_max(self):
        self.sw.alpha_max = 0.5
        try:
            alpha, n_iter = self.sw.find_step_length(None)
        except ValueError as err:
            self.assertEqual(
                str(err),
                'alpha_max must be positive and > alpha_init')
        else:
            self.fail('ValueError expected')

    def test_bad_curv_cond(self):
        self.sw.curv_cond = 1e-10
        try:
            alpha, n_iter = self.sw.find_step_length(None)
        except ValueError as err:
            self.assertEqual(
                str(err),
                'curv_cond must be suff_cond < curv_cond < 1')
        else:
            self.fail('ValueError expected')


if __name__ == "__main__":

    unittest.main()
