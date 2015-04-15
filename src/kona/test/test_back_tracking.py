import unittest

from kona.linalg.memory import KonaMemory
from kona.algorithms.util.linesearch import BackTracking
from kona.algorithms.util.merit import ObjectiveMerit

from kona.examples.simple_2by2 import Simple2x2

class BackTrackingTestCase(unittest.TestCase):


    def setUp(self):
        solver = Simple2x2()
        km = KonaMemory(solver)
        self.pf = pf = km.primal_factory
        self.sf = sf = km.state_factory

        pf.request_num_vectors(10)
        sf.request_num_vectors(10)

        self.merit = ObjectiveMerit(pf, sf)

        km.allocate_memory()

        self.bt = BackTracking() # leave all settings with initial values

        search_dir = self.pf.generate()
        search_dir._data.data[:] = [-1,0]

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
        self.bt.p_dot_dfdx = dfdx.inner(search_dir)


    def test_stops_after_one_iter(self):
        '''Assuming your first guess viloates sufficient decrease condition'''

        search_dir = self.pf.generate()
        search_dir._data.data[:] = [-1,0]

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

        self.bt.merit_function = self.merit
        self.bt.alpha_init = .3 #should evaluate 2.5, 2.5
        self.bt.rdtn_factor = .3
        self.bt.decr_cond = 1e-4
        alpha, n_iter = self.bt.find_step_length()

        self.assertEqual(n_iter, 1)
        self.assertEqual(alpha, .09)


    def test_stops_after_multiple_iter(self):
        '''Assuming your first guess viloates sufficient decrease condition'''

        search_dir = self.pf.generate()
        search_dir._data.data[:] = [-1,0]

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

        self.bt.merit_function = self.merit
        self.bt.alpha_init = 1
        self.bt.rdtn_factor = .95
        self.bt.decr_cond = 0.5
        alpha, n_iter = self.bt.find_step_length()

        self.assertEqual(n_iter, 3)


    def test_from_running_other_way(self):
        '''Check that it works when the search direction sign is flips'''

        search_dir = self.pf.generate()
        search_dir._data.data[:] = [4.25,0]

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
        self.bt.p_dot_dfdx = dfdx.inner(search_dir)

        self.merit.reset(search_dir, at_design, at_state, self.bt.p_dot_dfdx)

        self.bt.merit_function = self.merit
        self.bt.alpha_init = 1 #should evaluate 2.5, 2.5
        alpha, n_iter = self.bt.find_step_length()

        self.assertEqual(n_iter, 1)
        self.assertEqual(alpha, .3)

    def test_bad_search_direction(self):

        search_dir = self.pf.generate()
        search_dir._data.data[:] = [-1,0]

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

        self.bt.merit_function = self.merit
        self.bt.alpha_init = .3 #should evaluate 2.5, 2.5

        self.bt.p_dot_dfdx *= -1
        try:
            alpha, n_iter = self.bt.find_step_length()
        except ValueError as err:
            self.assertEqual(str(err), 'search direction is not a descent direction')
        else:
            self.fail('ValueError expected')

    def test_no_merit_function(self):
        self.bt.merit_function = None
        try:
            alpha, n_iter = self.bt.find_step_length()
        except ValueError as err:
            self.assertEqual(str(err), 'merit_function can not be None')
        else:
            self.fail('ValueError expected')

    def test_bad_alpha_init(self):
        self.bt.alpha_init = 1e6
        try:
            alpha, n_iter = self.bt.find_step_length()
        except ValueError as err:
            self.assertEqual(str(err), 'alpha_init must be 0 < alpha_init <=1')
        else:
            self.fail('ValueError expected')


if __name__ == "__main__":

    unittest.main()
