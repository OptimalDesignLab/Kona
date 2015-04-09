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


    def test_stops_after_one_iter(self):
        '''Assuming your first guess viloates sufficient decrease condition'''

        search_dir = self.pf.generate()
        search_dir.equals(1)

        at_design = self.pf.generate()
        at_design.equals(2)

        at_state = self.sf.generate()
        at_state.equals_primal_solution(at_design)

        state_work = self.sf.generate()

        adjoint = self.sf.generate()
        adjoint.equals_adjoint_solution(at_design, at_state, state_work)

        primal_work = self.pf.generate()

        dfdx = self.pf.generate()
        dfdx.equals_reduced_gradient(at_design, at_state, adjoint, primal_work)
        self.bt.p_dot_fx = dfdx.inner(search_dir)


        self.merit.reset(search_dir, at_design)

        self.bt.merit_function = self.merit.eval_func
        self.bt.alpha_init = .1 #should evaluate 2.5, 2.5
        alpha, n_iter = self.bt.find_step_length()

        self.assertEqual(n_iter, 1)

        #cheating to get value of function at 2.1, 2.1



    def test_stops_after_multiple_iter(self):
        '''Assuming your first guess viloates sufficient decrease condition'''
        # self.fail("untested")

    def test_from_left_to_right(self):
        '''Assuming your first guess viloates sufficient decrease condition'''
        # self.fail("untested")

    def test_from_right_to_left(self):
        '''Assuming your first guess viloates sufficient decrease condition'''
        # self.fail("untested")

if __name__ == "__main__":

    unittest.main()
