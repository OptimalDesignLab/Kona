import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from dummy_solver import DummySolver
from kona.linalg.matrices.common import dRdU, dCINdU
from kona.linalg.vectors.composite import CompositePrimalVector

class CompositePrimalVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 0, 5)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(2)
        km.state_factory.request_num_vectors(3)
        km.ineq_factory.request_num_vectors(3)
        km.allocate_memory()

        self.design = km.primal_factory.generate()
        self.state = km.state_factory.generate()
        self.adjoint = km.state_factory.generate()
        self.state_work = km.state_factory.generate()
        self.slack = km.ineq_factory.generate()
        self.dual_ineq = km.ineq_factory.generate()
        self.primal = CompositePrimalVector(self.design, self.slack)

        # set the evaluation point
        self.design.equals_init_design()
        self.state.equals_primal_solution(self.design)
        self.slack.equals(1.0)
        self.dual_ineq.equals(1.0)

        self.dv = km.primal_factory.generate()
        self.dv.base.data[:] = 2*np.ones(10)
        self.sv = km.ineq_factory.generate()
        self.sv.base.data[:] = 3*np.ones(5)
        self.pv = CompositePrimalVector(self.dv, self.sv)

    def test_bad_init_args(self):
        '''CompositePrimalVector test for bad initial arguments'''
        try:
            CompositePrimalVector(self.sv, self.sv)
        except TypeError as err:
            self.assertEqual(
                str(err),
                'CompositePrimalVector() >> ' +
                'Unidentified primal vector.')
        else:
            self.fail('TypeError expected')

        try:
            CompositePrimalVector(self.dv, self.dv)
        except TypeError as err:
            self.assertEqual(
                str(err),
                'CompositePrimalVector() >> ' +
                'Unidentified dual vector.')
        else:
            self.fail('TypeError expected')

    def test_barrier_not_set(self):
        '''CompositePrimalVector error for missing barrier term'''
        try:
            self.pv.equals_lagrangian_total_gradient(
                self.primal, self.state, self.dual_ineq, self.adjoint)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "CompositePrimalVector() >> Barrier factor must be set!")
        else:
            self.fail('AssertionError expected')

    def test_init_design(self):
        '''CompositeDualVector design initialization'''
        self.pv.equals_init_design()

        err = self.dv.base.data - 10*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0.0)

        err = self.sv.base.data - np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0.0)

    def test_lagrangian_tot_grad(self):
        '''CompositeDualVector lagrangian total gradient'''
        dCINdU(self.design, self.state).T.product(self.dual_ineq, self.adjoint)
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.plus(self.adjoint)
        self.state_work.times(-1.)
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.pv.barrier = 0.5
        self.pv.equals_lagrangian_total_gradient(
            self.primal, self.state, self.dual_ineq, self.adjoint)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 10. ** 2)
        self.assertEqual(self.pv.design.norm2, exp_dLdX_norm)
        exp_dLdS_norm = np.sqrt(5. * 0.5 ** 2)
        self.assertEqual(self.pv.slack.norm2, exp_dLdS_norm)


if __name__ == "__main__":
    unittest.main()
