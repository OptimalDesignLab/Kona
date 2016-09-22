import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from dummy_solver import DummySolver
from kona.linalg.matrices.common import *
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.vectors.composite import ReducedKKTVector

class ReducedKKTVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 5, 5)
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(5)
        km.state_factory.request_num_vectors(3)
        km.eq_factory.request_num_vectors(4)
        km.ineq_factory.request_num_vectors(8)
        km.allocate_memory()

        # get some work vectors
        self.primal_work = km.primal_factory.generate()
        self.state_work = km.state_factory.generate()
        self.eq_work = km.eq_factory.generate()
        self.slack_work = km.ineq_factory.generate()
        self.ineq_work = km.ineq_factory.generate()

        # get static eval point vectors
        self.design = km.primal_factory.generate()
        self.slack = km.ineq_factory.generate()
        self.primal = CompositePrimalVector(self.design, self.slack)
        self.state = km.state_factory.generate()
        self.adjoint = km.state_factory.generate()
        self.dual_eq = km.eq_factory.generate()
        self.dual_ineq = km.ineq_factory.generate()
        self.dual = CompositeDualVector(self.dual_eq, self.dual_ineq)
        self.at_kkt1 = ReducedKKTVector(self.primal, self.dual)
        self.at_kkt2 = ReducedKKTVector(self.primal, self.dual_ineq)
        self.at_kkt3 = ReducedKKTVector(self.design, self.dual_eq)

        # set the evaluation point
        self.design.equals_init_design()
        self.state.equals_primal_solution(self.design)
        self.slack.equals(1.0)
        self.dual.equals(1.0)

        # first KKT vector is a complete vector with design, slack, eq and ineq
        self.pv1 = km.primal_factory.generate()
        self.slack1 = km.ineq_factory.generate()
        self.eq1 = km.eq_factory.generate()
        self.ineq1 = km.ineq_factory.generate()
        self.pv1.base.data = 2*np.ones(10)
        self.slack1.base.data = 3*np.ones(5)
        primal1 = CompositePrimalVector(self.pv1, self.slack1)
        self.eq1.base.data = 4*np.ones(5)
        self.ineq1.base.data = 5*np.ones(5)
        dual1 = CompositeDualVector(self.eq1, self.ineq1)
        self.rkkt_vec1 = ReducedKKTVector(primal1, dual1)

        # second KKT vector has design, slack and ineq
        self.pv2 = km.primal_factory.generate()
        self.slack2 = km.ineq_factory.generate()
        self.ineq2 = km.ineq_factory.generate()
        self.pv2.base.data = 2*np.ones(10)
        self.slack2.base.data = 3*np.ones(5)
        self.ineq2.base.data = 4*np.ones(5)
        primal2 = CompositePrimalVector(self.pv2, self.slack2)
        dual2 = self.ineq2
        self.rkkt_vec2 = ReducedKKTVector(primal2, dual2)

        # third KKT vector only has design and eq
        self.pv3 = km.primal_factory.generate()
        self.eq3 = km.eq_factory.generate()
        primal3 = self.pv3
        dual3 = self.eq3
        self.rkkt_vec3 = ReducedKKTVector(primal3, dual3)

    def test_bad_init_args(self):

        try:
            ReducedKKTVector(self.eq1, self.eq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "ReducedKKTVector() >> Invalid primal vector. " +
                "Must be either DesignVector or CompositePrimalVector!")
        else:
            self.fail('AssertionError expected')

        try:
            ReducedKKTVector(self.pv1, self.pv1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "ReducedKKTVector() >> Mismatched dual vector. " +
                "Must be DualVectorEQ!")
        else:
            self.fail('AssertionError expected')

        try:
            ReducedKKTVector(
                CompositePrimalVector(self.pv1, self.ineq1), self.eq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "ReducedKKTVector() >> Mismatched dual vector. " +
                "Must be DualVecINEQ or CompositeDualVector!")
        else:
            self.fail('AssertionError expected')

    def test_initial_guess_case1(self):
        self.rkkt_vec1.equals_init_guess()

        err = self.pv1.base.data - 10 * np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.slack1.base.data - np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq1.base.data - self.rkkt_vec1.init_dual * (np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.ineq1.base.data - self.rkkt_vec1.init_dual * (np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)

    def test_init_guess_case2(self):
        self.rkkt_vec2.equals_init_guess()

        err = self.pv2.base.data - 10 * np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.slack2.base.data - np.ones(5)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.ineq2.base.data - self.rkkt_vec1.init_dual * (np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)

    def test_init_guess_case3(self):
        self.rkkt_vec3.equals_init_guess()

        err = self.pv3.base.data - 10*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq3.base.data - self.rkkt_vec2.init_dual*(np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)

    def test_kkt_conditions_case1(self):
        # case 1 has both equality and inequality constraints
        dCdU(self.design, self.state).T.product(self.dual, self.adjoint, self.state_work)
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.plus(self.adjoint)
        self.state_work.times(-1.)
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.rkkt_vec1.equals_KKT_conditions(
            self.at_kkt1, self.state, self.adjoint, 0.5)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 20.**2)
        self.assertEqual(self.rkkt_vec1.primal.design.norm2, exp_dLdX_norm)
        exp_dLdS_norm = np.sqrt(5. * 0.5**2)
        self.assertEqual(self.rkkt_vec1.primal.slack.norm2, exp_dLdS_norm)
        exp_dLdEq_norm = np.sqrt(5. * 200.**2)
        self.assertEqual(self.rkkt_vec1.dual.eq.norm2, exp_dLdEq_norm)
        exp_dLdIn_norm = np.sqrt(5. * 199.**2)
        self.assertEqual(self.rkkt_vec1.dual.ineq.norm2, exp_dLdIn_norm)

    def test_kkt_conditions_case2(self):
        # case 2 has only inequality constraints
        dCINdU(self.design, self.state).T.product(self.dual_ineq, self.adjoint)
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.plus(self.adjoint)
        self.state_work.times(-1.)
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.rkkt_vec2.equals_KKT_conditions(
            self.at_kkt2, self.state, self.adjoint, 0.5)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 10. ** 2)
        self.assertEqual(self.rkkt_vec2.primal.design.norm2, exp_dLdX_norm)
        exp_dLdS_norm = np.sqrt(5. * 0.5**2)
        self.assertEqual(self.rkkt_vec2.primal.slack.norm2, exp_dLdS_norm)
        exp_dLdIn_norm = np.sqrt(5. * 199. ** 2)
        self.assertEqual(self.rkkt_vec2.dual.norm2, exp_dLdIn_norm)

    def test_kkt_conditions_case3(self):
        # case 3 has only equality constraints
        dCEQdU(self.design, self.state).T.product(self.dual_eq, self.adjoint)
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.plus(self.adjoint)
        self.state_work.times(-1.)
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.rkkt_vec3.equals_KKT_conditions(
            self.at_kkt3, self.state, self.adjoint, 0.0)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 10. ** 2)
        self.assertEqual(self.rkkt_vec3.primal.norm2, exp_dLdX_norm)
        exp_dLdEq_norm = np.sqrt(5. * 200. ** 2)
        self.assertEqual(self.rkkt_vec3.dual.norm2, exp_dLdEq_norm)


if __name__ == "__main__":
    unittest.main()
