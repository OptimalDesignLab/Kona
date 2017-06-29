import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from dummy_solver import DummySolver
from kona.linalg.matrices.common import *
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.vectors.composite import PrimalDualVector

class PrimalDualVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 5, 5)  # 10 DVs, 5 equality, 5 inequality
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(5)
        km.state_factory.request_num_vectors(3)
        km.eq_factory.request_num_vectors(4)
        km.ineq_factory.request_num_vectors(8)  # doe we need this many?
        km.allocate_memory()

        # get some work vectors
        self.primal_work = km.primal_factory.generate()
        self.slack_work = km.ineq_factory.generate()  # only used for error testing
        self.state_work = km.state_factory.generate()
        self.eq_work = km.eq_factory.generate()
        self.ineq_work = km.ineq_factory.generate()

        # get static eval point vectors
        self.design = km.primal_factory.generate()
        self.state = km.state_factory.generate()
        self.adjoint = km.state_factory.generate()
        self.dual_eq = km.eq_factory.generate()
        self.dual_ineq = km.ineq_factory.generate()
        self.dual = CompositeDualVector(self.dual_eq, self.dual_ineq)
        self.at_kkt1 = PrimalDualVector(self.design, self.dual)
        self.at_kkt2 = PrimalDualVector(self.design, self.dual_ineq)
        self.at_kkt3 = PrimalDualVector(self.design, self.dual_eq)

        # set the evaluation point
        self.design.equals_init_design()
        self.state.equals_primal_solution(self.design)
        self.dual.equals(1.0)

        # case 1: primal-dual vector is a complete vector with design, eq and ineq
        self.pv1 = km.primal_factory.generate()
        self.eq1 = km.eq_factory.generate()
        self.ineq1 = km.ineq_factory.generate()
        self.pv1.base.data = 2*np.ones(10)
        self.eq1.base.data = 4*np.ones(5)
        self.ineq1.base.data = 5*np.ones(5)
        primal1 = self.pv1
        dual1 = CompositeDualVector(self.eq1, self.ineq1)
        self.pd_vec1 = PrimalDualVector(primal1, dual1)

        # case 2: primal-dual vector has design and ineq
        self.pv2 = km.primal_factory.generate()
        self.ineq2 = km.ineq_factory.generate()
        self.pv2.base.data = 2*np.ones(10)
        self.ineq2.base.data = 4*np.ones(5)
        primal2 = self.pv2
        dual2 = self.ineq2
        self.pd_vec2 = PrimalDualVector(primal2, dual2)

        # case 3: primal-dual vector only has design and eq
        self.pv3 = km.primal_factory.generate()
        self.eq3 = km.eq_factory.generate()
        primal3 = self.pv3
        dual3 = self.eq3
        self.pd_vec3 = PrimalDualVector(primal3, dual3)

    def test_bad_init_args(self):
        try:
            PrimalDualVector(self.eq1, self.eq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> Mismatched primal vector. " +
                "Must be DesignVector!")
        else:
            self.fail('AssertionError expected')

        try:
            PrimalDualVector(self.pv1, self.pv1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> Mismatched dual vector. " +
                "Must be DualVectorEQ, DualVectorINEQ CompositeDualVector!")
        else:
            self.fail('AssertionError expected')

        try:
            PrimalDualVector(self.ineq1, self.ineq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> Mismatched primal vector. " +
                "Must be DesignVector!")
        else:
            self.fail('AssertionError expected')

        try:
            PrimalDualVector(
                CompositePrimalVector(self.pv1, self.slack_work), self.eq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> Mismatched primal vector. " +
                "Must be DesignVector!")
        else:
            self.fail('AssertionError expected')

    def test_initial_guess_case1(self):
        self.pd_vec1.equals_init_guess()

        err = self.pv1.base.data - 10 * np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq1.base.data - self.pd_vec1.init_dual * (np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.ineq1.base.data - self.pd_vec1.init_dual * (np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)

    def test_init_guess_case2(self):
        self.pd_vec2.equals_init_guess()

        err = self.pv2.base.data - 10 * np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.ineq2.base.data - self.pd_vec2.init_dual * (np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)

    def test_init_guess_case3(self):
        self.pd_vec3.equals_init_guess()

        err = self.pv3.base.data - 10*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

        err = self.eq3.base.data - self.pd_vec3.init_dual*(np.ones(5))
        self.assertEqual(np.linalg.norm(err), 0)
