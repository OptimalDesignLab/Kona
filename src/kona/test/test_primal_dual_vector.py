import unittest
import numpy as np

from kona.linalg.memory import KonaMemory
from dummy_solver import DummySolver
from kona.linalg.matrices.common import *
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.vectors.composite import PrimalDualVector

class PrimalDualVectorTestCase(unittest.TestCase):

    def setUp(self):
        solver = DummySolver(10, 5, 5)  # 10 DVs, 5 equality, 5 inequality
        self.km = km = KonaMemory(solver)

        km.primal_factory.request_num_vectors(11)
        km.state_factory.request_num_vectors(3)
        km.eq_factory.request_num_vectors(7)
        km.ineq_factory.request_num_vectors(11)  # doe we need this many?
        km.allocate_memory()

        # get some work vectors
        self.primal_work1 = km.primal_factory.generate()
        self.primal_work2 = km.primal_factory.generate()
        self.primal_work3 = km.primal_factory.generate()
        self.slack_work = km.ineq_factory.generate()  # only used for error testing
        self.state_work = km.state_factory.generate()
        self.eq_work1 = km.eq_factory.generate()
        self.eq_work2 = km.eq_factory.generate()
        self.eq_work3 = km.eq_factory.generate()
        self.ineq_work1 = km.ineq_factory.generate()
        self.ineq_work2 = km.ineq_factory.generate()
        self.ineq_work3 = km.ineq_factory.generate()

        # get static eval point vectors
        self.design = km.primal_factory.generate()
        self.state = km.state_factory.generate()
        self.adjoint = km.state_factory.generate()
        self.dual_eq = km.eq_factory.generate()
        self.dual_ineq = km.ineq_factory.generate()
        self.at_pd1 = PrimalDualVector(self.design, eq_vec=self.dual_eq,
                                       ineq_vec=self.dual_ineq)
        self.at_pd2 = PrimalDualVector(self.design, ineq_vec=self.dual_ineq)
        self.at_pd3 = PrimalDualVector(self.design, eq_vec=self.dual_eq)
        self.at_pd4 = PrimalDualVector(self.design)

        # set the evaluation point
        self.design.equals_init_design()
        self.state.equals_primal_solution(self.design)
        self.dual_eq.equals(1.0)
        self.dual_ineq.equals(1.0)

        # case 1: primal-dual vector is a complete vector with design, eq and ineq
        self.pv1 = km.primal_factory.generate()
        self.eq1 = km.eq_factory.generate()
        self.ineq1 = km.ineq_factory.generate()
        self.pv1.base.data = 2*np.ones(10)
        self.eq1.base.data = 4*np.ones(5)
        self.ineq1.base.data = 5*np.ones(5)
        primal1 = self.pv1
        dual_eq1 = self.eq1
        dual_ineq1 = self.ineq1
        self.pd_vec1 = PrimalDualVector(primal1, eq_vec=dual_eq1, ineq_vec=dual_ineq1)
        self.pd_work11 = PrimalDualVector(self.primal_work1, eq_vec=self.eq_work1,
                                          ineq_vec=self.ineq_work1)
        self.pd_work12 = PrimalDualVector(self.primal_work2, eq_vec=self.eq_work2,
                                          ineq_vec=self.ineq_work2)
        self.pd_work13 = PrimalDualVector(self.primal_work3, eq_vec=self.eq_work3,
                                          ineq_vec=self.ineq_work3)

        # case 2: primal-dual vector has design and ineq
        self.pv2 = km.primal_factory.generate()
        self.ineq2 = km.ineq_factory.generate()
        self.pv2.base.data = 2*np.ones(10)
        self.ineq2.base.data = 4*np.ones(5)
        primal2 = self.pv2
        dual2 = self.ineq2
        self.pd_vec2 = PrimalDualVector(primal2, ineq_vec=dual2)
        self.pd_work21 = PrimalDualVector(self.primal_work1, ineq_vec=self.ineq_work1)
        self.pd_work22 = PrimalDualVector(self.primal_work2, ineq_vec=self.ineq_work2)

        # case 3: primal-dual vector only has design and eq
        self.pv3 = km.primal_factory.generate()
        self.eq3 = km.eq_factory.generate()
        primal3 = self.pv3
        dual3 = self.eq3
        self.pd_vec3 = PrimalDualVector(primal3, eq_vec=dual3)
        self.pd_work31 = PrimalDualVector(self.primal_work1, eq_vec=self.eq_work1)
        self.pd_work32 = PrimalDualVector(self.primal_work2, eq_vec=self.eq_work2)

        # case 4: primal-dual vector has design only
        self.pv4 = km.primal_factory.generate()
        primal4 = self.pv4
        self.pd_vec4 = PrimalDualVector(primal4)
        self.pd_work41 = PrimalDualVector(self.primal_work1)
        self.pd_work42 = PrimalDualVector(self.primal_work2)

    def test_bad_init_args(self):
        try:
            PrimalDualVector(self.eq1, self.eq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> primal_vec must be a DesignVector!")
        else:
            self.fail('AssertionError expected')

        try:
            PrimalDualVector(self.pv1, self.pv1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> eq_vec must be a DualVectorEQ!")
        else:
            self.fail('AssertionError expected')

        try:
            PrimalDualVector(self.ineq1, self.ineq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> primal_vec must be a DesignVector!")
        else:
            self.fail('AssertionError expected')

        try:
            PrimalDualVector(self.pv1, self.eq1, self.eq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> ineq_vec must be a DualVectorINEQ!")
        else:
            self.fail('AssertionError expected')

        try:
            PrimalDualVector(
                CompositePrimalVector(self.pv1, self.slack_work), self.eq1)
        except AssertionError as err:
            self.assertEqual(
                str(err),
                "PrimalDualVector() >> primal_vec must be a DesignVector!")
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

    def test_init_guess_case4(self):
        self.pd_vec4.equals_init_guess()

        err = self.pv4.base.data - 10*np.ones(10)
        self.assertEqual(np.linalg.norm(err), 0)

    def test_get_num_var_case1(self):
        self.assertEqual(self.pd_vec1.get_num_var(), 20)

    def test_get_num_var_case2(self):
        self.assertEqual(self.pd_vec2.get_num_var(), 15)

    def test_get_num_var_case3(self):
        self.assertEqual(self.pd_vec3.get_num_var(), 15)

    def test_get_num_var_case4(self):
        self.assertEqual(self.pd_vec4.get_num_var(), 10)

    def test_get_dual_case1(self):
        dual = self.pd_vec1.get_dual()
        self.assertTrue(isinstance(dual, CompositeDualVector))

    def test_get_dual_case2(self):
        dual = self.pd_vec2.get_dual()
        self.assertTrue(isinstance(dual, DualVectorINEQ))

    def test_get_dual_case3(self):
        dual = self.pd_vec3.get_dual()
        self.assertTrue(isinstance(dual, DualVectorEQ))

    def test_get_dual_case4(self):
        dual = self.pd_vec4.get_dual()
        self.assertTrue(dual is None)

    def test_kkt_conditions_case1(self):
        # case 1 has both equality and inequality constraints
        # recall: self.dual = 1, so dCeqdU^T*dual.eq + dCineqdU^Tdual.ineq = 5 + 5 = 10
        dCdU(self.design, self.state).T.product(CompositeDualVector(self.dual_eq, self.dual_ineq),
                                                self.adjoint, self.state_work)
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.minus(self.adjoint) # L = f - ceq^T*lambda_eq - cineq^T*lambda_ineq
        self.state_work.times(-1.)
        # We get adjoint = -11
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.pd_vec1.equals_KKT_conditions(self.at_pd1, self.state, self.adjoint)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 20.**2)
        self.assertEqual(self.pd_vec1.primal.norm2, exp_dLdX_norm)
        exp_dLdEq_norm = np.sqrt(5. * 200.**2)
        self.assertEqual(self.pd_vec1.eq.norm2, exp_dLdEq_norm)
        exp_dLdIn_norm = np.sqrt(5. * 200**2)
        self.assertEqual(self.pd_vec1.ineq.norm2, exp_dLdIn_norm)

    def test_kkt_conditions_case2(self):
        # case 2 has inequality constraints only
        # recall: self.dual = 1, so dCineqdU^Tdual.ineq = 5 = 5
        dCdU(self.design, self.state).T.product(self.dual_ineq, self.adjoint, self.state_work)
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.minus(self.adjoint) # L = f - cineq^T*lambda_ineq
        self.state_work.times(-1.)
        # We get adjoint = -6
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.pd_vec2.equals_KKT_conditions(self.at_pd2, self.state, self.adjoint)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 10.**2)
        self.assertEqual(self.pd_vec2.primal.norm2, exp_dLdX_norm)
        exp_dLdEq_norm = np.sqrt(5. * 200.**2)
        self.assertEqual(self.pd_vec2.ineq.norm2, exp_dLdEq_norm)

    def test_kkt_conditions_case3(self):
        # case 3 has equality constraints only
        # recall: self.dual = 1, so dCeqdU^Tdual.eq = 5 = 5
        dCdU(self.design, self.state).T.product(self.dual_eq, self.adjoint, self.state_work)
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.minus(self.adjoint) # L = f - cineq^T*lambda_ineq
        self.state_work.times(-1.)
        # We get adjoint = -6
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.pd_vec3.equals_KKT_conditions(self.at_pd3, self.state, self.adjoint)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 10.**2)
        self.assertEqual(self.pd_vec3.primal.norm2, exp_dLdX_norm)
        exp_dLdEq_norm = np.sqrt(5. * 200**2)
        self.assertEqual(self.pd_vec3.eq.norm2, exp_dLdEq_norm)

    def test_kkt_conditions_case4(self):
        # case 4 has no constraints
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.times(-1.)
        # We get adjoint = -1
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        self.pd_vec4.equals_KKT_conditions(self.at_pd4, self.state, self.adjoint)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 0.**2)
        self.assertEqual(self.pd_vec4.primal.norm2, exp_dLdX_norm)

    def test_homotopy_residual_case1(self):
        # case 1 has both equality and inequality constraints
        # recall: self.dual = 1, so dCeqdU^T*dual.eq + dCineqdU^Tdual.ineq = 5 + 5 = 10
        dCdU(self.design, self.state).T.product(CompositeDualVector(self.dual_eq, self.dual_ineq),
                                                self.adjoint, self.state_work)
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.minus(self.adjoint) # L = f - ceq^T*lambda_eq - cineq^T*lambda_ineq
        self.state_work.times(-1.)
        # We get adjoint = -11
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        dLdx = self.pd_work11
        dLdx.equals_KKT_conditions(self.at_pd1, self.state, self.adjoint)

        init = self.pd_work12
        init.equals_init_guess()
        init.eq.equals_constraints(init.primal, self.state)
        init.ineq.equals_constraints(init.primal, self.state)

        x = self.at_pd1
        self.pd_vec1.equals_homotopy_residual(dLdx, x, init, mu=0.5)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 10.**2)
        self.assertAlmostEqual(self.pd_vec1.primal.norm2, exp_dLdX_norm, places=10)
        exp_dLdEq_norm = np.sqrt(5. * 100.5**2)
        self.assertAlmostEqual(self.pd_vec1.eq.norm2, exp_dLdEq_norm, places=10)
        exp_dLdIn_norm = np.sqrt(5. * 29701.95**2)
        self.assertAlmostEqual(self.pd_vec1.ineq.norm2, exp_dLdIn_norm, places=10)

        # test get_optimality_and_feasiblity while we are at it
        opt, feas = self.pd_vec1.get_optimality_and_feasiblity()
        self.assertAlmostEqual(opt, exp_dLdX_norm, places=10)
        self.assertAlmostEqual(feas, np.sqrt(exp_dLdEq_norm**2 + exp_dLdIn_norm**2), places=10)

    def test_homotopy_residual_case2(self):
        # case 2 has inequality constraints only
        # recall: self.dual = 1, so dCineqdU^Tdual.ineq = 5
        dCdU(self.design, self.state).T.product(self.dual_ineq, self.adjoint, self.state_work)
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.minus(self.adjoint) # L = f - cineq^T*lambda_ineq
        self.state_work.times(-1.)
        # We get adjoint = -6
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        dLdx = self.pd_work21
        dLdx.equals_KKT_conditions(self.at_pd2, self.state, self.adjoint)
        init = self.pd_work22
        init.equals_init_guess()
        init.ineq.equals_constraints(init.primal, self.state)

        x = self.at_pd2
        self.pd_vec2.equals_homotopy_residual(dLdx, x, init, mu=0.5)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 5**2)
        self.assertAlmostEqual(self.pd_vec2.primal.norm2, exp_dLdX_norm, places=10)
        exp_dLdIn_norm = np.sqrt(5. * 29701.95**2)
        self.assertAlmostEqual(self.pd_vec2.ineq.norm2, exp_dLdIn_norm, places=10)

        # test get_optimality_and_feasiblity while we are at it
        opt, feas = self.pd_vec2.get_optimality_and_feasiblity()
        self.assertAlmostEqual(opt, exp_dLdX_norm, places=10)
        self.assertAlmostEqual(feas, exp_dLdIn_norm, places=10)

    def test_homotopy_residual_case3(self):
        # case 3 has equality constraints only
        # recall: self.dual = 1, so dCeqdU^Tdual.ineq = 5
        dCdU(self.design, self.state).T.product(self.dual_eq, self.adjoint, self.state_work)
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.minus(self.adjoint) # L = f - ceq^T*lambda_eq
        self.state_work.times(-1.)
        # We get adjoint = -6
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        dLdx = self.pd_work31
        dLdx.equals_KKT_conditions(self.at_pd3, self.state, self.adjoint)
        init = self.pd_work32
        init.equals_init_guess()
        init.eq.equals_constraints(init.primal, self.state)

        x = self.at_pd3
        self.pd_vec3.equals_homotopy_residual(dLdx, x, init, mu=0.5)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 5**2)
        self.assertAlmostEqual(self.pd_vec3.primal.norm2, exp_dLdX_norm, places=10)
        exp_dLdEq_norm = np.sqrt(5. * 100.5**2)
        self.assertAlmostEqual(self.pd_vec3.eq.norm2, exp_dLdEq_norm, places=10)

        # test get_optimality_and_feasiblity while we are at it
        opt, feas = self.pd_vec3.get_optimality_and_feasiblity()
        self.assertAlmostEqual(opt, exp_dLdX_norm, places=10)
        self.assertAlmostEqual(feas, exp_dLdEq_norm, places=10)

    def test_homotopy_residual_case4(self):
        # case 4 has no constraints
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.times(-1.)
        # We get adjoint = -1
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        dLdx = self.pd_work41
        dLdx.equals_KKT_conditions(self.at_pd4, self.state, self.adjoint)
        init = self.pd_work42
        init.equals_init_guess()

        x = self.at_pd4
        self.pd_vec4.equals_homotopy_residual(dLdx, x, init, mu=0.5)

        # check results
        exp_dLdX_norm = np.sqrt(10. * 0**2)
        self.assertAlmostEqual(self.pd_vec3.primal.norm2, exp_dLdX_norm, places=10)

        # test get_optimality_and_feasiblity while we are at it
        opt, feas = self.pd_vec4.get_optimality_and_feasiblity()
        self.assertAlmostEqual(opt, exp_dLdX_norm, places=10)
        self.assertAlmostEqual(feas, 0.0, places=10)

    def test_equals_predictor_rhs_case1(self):
        # case 1 has both equality and inequality constraints
        # recall: self.dual = 1, so dCeqdU^T*dual.eq + dCineqdU^Tdual.ineq = 5 + 5 = 10
        dCdU(self.design, self.state).T.product(CompositeDualVector(self.dual_eq, self.dual_ineq),
                                                self.adjoint, self.state_work)
        # recall: dFdU = -1
        self.state_work.equals_objective_partial(self.design, self.state)
        self.state_work.minus(self.adjoint) # L = f - ceq^T*lambda_eq - cineq^T*lambda_ineq
        self.state_work.times(-1.)
        # We get adjoint = -11
        dRdU(self.design, self.state).T.solve(self.state_work, self.adjoint)
        dLdx = self.pd_work11
        dLdx.equals_KKT_conditions(self.at_pd1, self.state, self.adjoint)

        init = self.pd_work12
        init.equals_init_guess()
        init.eq.equals_constraints(init.primal, self.state)
        init.ineq.equals_constraints(init.primal, self.state)

        x = self.at_pd1
        self.pd_vec1.equals_homotopy_residual(dLdx, x, init, mu=0.5)
        dmu = 1e-7
        self.pd_work13.equals_homotopy_residual(dLdx, x, init, mu=0.5+dmu)
        self.pd_work13.minus(self.pd_vec1)
        self.pd_work13.divide_by(dmu)
        self.pd_vec1.equals_predictor_rhs(dLdx, x, init, mu=0.5)

        # check results
        self.assertAlmostEqual(self.pd_vec1.primal.norm2, self.pd_work13.primal.norm2, places=5)
        self.assertAlmostEqual(self.pd_vec1.eq.norm2/self.pd_work13.eq.norm2, 1.0, places=5)
        self.assertAlmostEqual(self.pd_vec1.ineq.norm2/self.pd_work13.ineq.norm2, 1.0, places=5)

    def test_get_base_data_case1(self):
        # case 1 has both equality and inequality constraints
        A = np.zeros((10+5+5,1))
        self.at_pd1.get_base_data(A[:,0])
        B = np.ones_like(A)
        B[0:10] *= 10
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_get_base_data_case2(self):
        # case 2 has inequality constraints only
        A = np.zeros((10+5,1))
        self.at_pd2.get_base_data(A[:,0])
        B = np.ones_like(A)
        B[0:10] *= 10
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_get_base_data_case3(self):
        # case 3 has equality constraints only
        A = np.zeros((10+5,1))
        self.at_pd3.get_base_data(A[:,0])
        B = np.ones_like(A)
        B[0:10] *= 10
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_get_base_data_case4(self):
        # case 4 has no constraints
        A = np.zeros((10,1))
        print A.shape
        self.at_pd4.get_base_data(A[:,0])
        B = np.ones_like(A)
        B[0:10] *= 10
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_set_base_data_case1(self):
        # case 1 has both equality and inequality constraints
        A = np.random.random((10+5+5,1))
        self.pd_vec1.set_base_data(A[:,0])
        B = np.zeros_like(A)
        self.pd_vec1.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_set_base_data_case2(self):
        # case 2 has only inequality constraints
        A = np.random.random((10+5,1))
        self.pd_vec2.set_base_data(A[:,0])
        B = np.zeros_like(A)
        self.pd_vec2.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_set_base_data_case3(self):
        # case 3 has only equality constraints
        A = np.random.random((10+5,1))
        self.pd_vec3.set_base_data(A[:,0])
        B = np.zeros_like(A)
        self.pd_vec3.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_set_base_data_case4(self):
        # case 4 has no constraints
        A = np.random.random((10,1))
        self.pd_vec4.set_base_data(A[:,0])
        B = np.zeros_like(A)
        self.pd_vec4.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])
