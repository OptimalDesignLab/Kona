import unittest
from collections import deque

import numpy as np

from kona.linalg.memory import KonaMemory
from kona.user import UserSolver
from kona.linalg.vectors.composite import PrimalDualVector
from kona.linalg.matrices.hessian import AndersonMultiSecant

class AndersonMultiSecantTestCase(unittest.TestCase):
    '''Test cases for Anderson-Acceleration multi-secant method'''

    def test_generate_vector_case1(self):
        # case 1: unconstrained
        max_stored = 3
        solver = UserSolver(3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant(pf, optns)
        km.allocate_memory()

        vec = aa._generate_vector()
        self.assertTrue(isinstance(vec, PrimalDualVector))
        self.assertTrue(vec.primal is not None)
        self.assertTrue(vec.eq is None)
        self.assertTrue(vec.ineq is None)

    def test_generate_vector_case2(self):
        # case 2: constrained with equality constraints only
        max_stored = 3
        solver = UserSolver(10, num_eq=3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf], optns)
        km.allocate_memory()

        vec = aa._generate_vector()
        self.assertTrue(isinstance(vec, PrimalDualVector))
        self.assertTrue(vec.primal is not None)
        self.assertTrue(vec.eq is not None)
        self.assertTrue(vec.ineq is None)

    def test_generate_vector_case3(self):
        # case 3: constrained with inequality constraints only
        max_stored = 3
        solver = UserSolver(10, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, ineqf], optns)
        km.allocate_memory()

        vec = aa._generate_vector()
        self.assertTrue(isinstance(vec, PrimalDualVector))
        self.assertTrue(vec.primal is not None)
        self.assertTrue(vec.eq is None)
        self.assertTrue(vec.ineq is not None)

    def test_generate_vector_case4(self):
        # case 4: constrained with both equality and inequality constraints
        max_stored = 3
        solver = UserSolver(10, num_eq=3, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf, ineqf], optns)
        km.allocate_memory()

        vec = aa._generate_vector()
        self.assertTrue(isinstance(vec, PrimalDualVector))
        self.assertTrue(vec.primal is not None)
        self.assertTrue(vec.eq is not None)
        self.assertTrue(vec.ineq is not None)

    def test_set_initial_data_case1(self):
        # case 1: unconstrained
        max_stored = 3
        solver = UserSolver(3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant(pf, optns)
        # add request that will be used to create pd_vec
        km.primal_factory.request_num_vectors(1)
        km.allocate_memory()

        p_vec = km.primal_factory.generate()
        pd_vec = PrimalDualVector(p_vec)
        A = np.random.random((3,1))
        pd_vec.set_base_data(A[:,0])
        aa.set_initial_data(pd_vec)
        B = np.zeros_like(A)
        aa.init.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_set_initial_data_case2(self):
        # case 2: constrained with equality constraints only
        max_stored = 3
        solver = UserSolver(10, num_eq=3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf], optns)
        # add request that will be used to create pd_vec
        km.primal_factory.request_num_vectors(1)
        km.eq_factory.request_num_vectors(1)
        km.allocate_memory()

        p_vec = km.primal_factory.generate()
        eq_vec = km.eq_factory.generate()
        pd_vec = PrimalDualVector(p_vec, eq_vec=eq_vec)
        A = np.random.random((13,1))
        pd_vec.set_base_data(A[:,0])
        aa.set_initial_data(pd_vec)
        B = np.zeros_like(A)
        aa.init.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_set_initial_data_case3(self):
        # case 3: constrained with inequality constraints only
        max_stored = 3
        solver = UserSolver(10, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, ineqf], optns)
        # add request that will be used to create pd_vec
        km.primal_factory.request_num_vectors(1)
        km.ineq_factory.request_num_vectors(1)
        km.allocate_memory()

        p_vec = km.primal_factory.generate()
        ineq_vec = km.ineq_factory.generate()
        pd_vec = PrimalDualVector(p_vec, ineq_vec=ineq_vec)
        A = np.random.random((15,1))
        pd_vec.set_base_data(A[:,0])
        aa.set_initial_data(pd_vec)
        B = np.zeros_like(A)
        aa.init.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_set_initial_data_case4(self):
        # case 4: constrained with both equality and inequality constraints
        max_stored = 3
        solver = UserSolver(10, num_eq=3, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf, ineqf], optns)
        # add request that will be used to create pd_vec
        km.primal_factory.request_num_vectors(1)
        km.eq_factory.request_num_vectors(1)
        km.ineq_factory.request_num_vectors(1)
        km.allocate_memory()

        p_vec = km.primal_factory.generate()
        eq_vec = km.eq_factory.generate()
        ineq_vec = km.ineq_factory.generate()
        pd_vec = PrimalDualVector(p_vec, eq_vec=eq_vec, ineq_vec=ineq_vec)
        A = np.random.random((18,1))
        pd_vec.set_base_data(A[:,0])
        aa.set_initial_data(pd_vec)
        B = np.zeros_like(A)
        aa.init.get_base_data(B[:,0])
        for i in range(A.shape[0]):
            self.assertEqual(A[i], B[i])

    def test_add_to_history_case1(self):
        # case 1: unconstrained
        max_stored = 3
        solver = UserSolver(3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant(pf, optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        x = PrimalDualVector(p_vec1)
        p_vec2 = km.primal_factory.generate()
        r = PrimalDualVector(p_vec2)

        # Create synthetic solution and residual data and feed into aa history
        Xrand = np.random.random((3,max_stored+1))
        Rrand = np.random.random((3,max_stored+1))
        for k in range(max_stored+1):
            x.set_base_data(Xrand[:,k])
            r.set_base_data(Rrand[:,k])
            aa.add_to_history(x, r)
        # At this point, the first vector from Xrand and Rrand should have been discardded
        A = np.zeros(Xrand.shape[0])
        for k in range(max_stored):
            aa.x_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Xrand[i,k+1])
            aa.r_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Rrand[i,k+1])

    def test_add_to_history_case2(self):
        # case 2: constrained with equality constraints only
        max_stored = 3
        solver = UserSolver(10, num_eq=3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf], optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.eq_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        eq_vec1 = km.eq_factory.generate()
        x = PrimalDualVector(p_vec1, eq_vec=eq_vec1)
        p_vec2 = km.primal_factory.generate()
        eq_vec2 = km.eq_factory.generate()
        r = PrimalDualVector(p_vec2, eq_vec=eq_vec2)

        # Create synthetic solution and residual data and feed into aa history
        Xrand = np.random.random((13,max_stored+1))
        Rrand = np.random.random((13,max_stored+1))
        for k in range(max_stored+1):
            x.set_base_data(Xrand[:,k])
            r.set_base_data(Rrand[:,k])
            aa.add_to_history(x, r)
        # At this point, the first vector from Xrand and Rrand should have been discardded
        A = np.zeros(Xrand.shape[0])
        for k in range(max_stored):
            aa.x_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Xrand[i,k+1])
            aa.r_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Rrand[i,k+1])

    def test_add_to_history_case3(self):
        # case 3: constrained with inequality constraints only
        max_stored = 3
        solver = UserSolver(10, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, ineqf], optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.ineq_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        ineq_vec1 = km.ineq_factory.generate()
        x = PrimalDualVector(p_vec1, ineq_vec=ineq_vec1)
        p_vec2 = km.primal_factory.generate()
        ineq_vec2 = km.ineq_factory.generate()
        r = PrimalDualVector(p_vec2, ineq_vec=ineq_vec2)

        # Create synthetic solution and residual data and feed into aa history
        Xrand = np.random.random((15,max_stored+1))
        Rrand = np.random.random((15,max_stored+1))
        for k in range(max_stored+1):
            x.set_base_data(Xrand[:,k])
            r.set_base_data(Rrand[:,k])
            aa.add_to_history(x, r)
        # At this point, the first vector from Xrand and Rrand should have been discardded
        A = np.zeros(Xrand.shape[0])
        for k in range(max_stored):
            aa.x_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Xrand[i,k+1])
            aa.r_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Rrand[i,k+1])

    def test_add_to_history_case4(self):
        # case 4: constrained with both equality and inequality constraints
        max_stored = 3
        solver = UserSolver(10, num_eq=3, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf, ineqf], optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.eq_factory.request_num_vectors(2)
        km.ineq_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        eq_vec1 = km.eq_factory.generate()
        ineq_vec1 = km.ineq_factory.generate()
        x = PrimalDualVector(p_vec1, eq_vec=eq_vec1, ineq_vec=ineq_vec1)
        p_vec2 = km.primal_factory.generate()
        eq_vec2 = km.eq_factory.generate()
        ineq_vec2 = km.ineq_factory.generate()
        r = PrimalDualVector(p_vec2, eq_vec=eq_vec2, ineq_vec=ineq_vec2)

        # Create synthetic solution and residual data and feed into aa history
        Xrand = np.random.random((18,max_stored+1))
        Rrand = np.random.random((18,max_stored+1))
        for k in range(max_stored+1):
            x.set_base_data(Xrand[:,k])
            r.set_base_data(Rrand[:,k])
            aa.add_to_history(x, r)
        # At this point, the first vector from Xrand and Rrand should have been discardded
        A = np.zeros(Xrand.shape[0])
        for k in range(max_stored):
            aa.x_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Xrand[i,k+1])
            aa.r_hist[k].get_base_data(A[:])
            for i in range(Xrand.shape[0]):
                self.assertEqual(A[i],Rrand[i,k+1])

    def test_build_difference_matrices_case1(self):
        # case 1: unconstrained
        max_stored = 3
        solver = UserSolver(3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant(pf, optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        x = PrimalDualVector(p_vec1)
        p_vec2 = km.primal_factory.generate()
        r = PrimalDualVector(p_vec2)
        aa.set_initial_data(x)

        # Create synthetic solution and residual data and feed into aa history
        x_scalar = np.linspace(1, max_stored, num=max_stored)
        r_scalar = np.ones_like(x_scalar)
        for k in range(max_stored):
            x.equals(x_scalar[k])
            r.equals(r_scalar[k])
            aa.add_to_history(x, r)
        aa.build_difference_matrices()
        A = np.zeros(3)
        for k in range(max_stored-1):
            aa.x_diff[k].get_base_data(A[:])
            for i in range(A.shape[0]):
                self.assertAlmostEqual(A[i], 1.0, places=14)

    def test_build_difference_matrices_case2(self):
        # case 2: constrained with equality constraints only
        max_stored = 3
        solver = UserSolver(10, num_eq=3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf], optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.eq_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        eq_vec1 = km.eq_factory.generate()
        x = PrimalDualVector(p_vec1, eq_vec=eq_vec1)
        p_vec2 = km.primal_factory.generate()
        eq_vec2 = km.eq_factory.generate()
        r = PrimalDualVector(p_vec2, eq_vec=eq_vec2)
        aa.set_initial_data(x)

        # Create synthetic solution and residual data and feed into aa history
        x_scalar = np.linspace(1, max_stored, num=max_stored)
        r_scalar = np.ones_like(x_scalar)
        for k in range(max_stored):
            x.equals(x_scalar[k])
            r.equals(r_scalar[k])
            aa.add_to_history(x, r)
        aa.build_difference_matrices()
        A = np.zeros(13)
        for k in range(max_stored-1):
            aa.x_diff[k].get_base_data(A[:])
            for i in range(A.shape[0]):
                self.assertAlmostEqual(A[i], 1.0, places=14)

    def test_build_difference_matrices_case3(self):
        # case 3: constrained with inequality constraints only
        max_stored = 3
        solver = UserSolver(10, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, ineqf], optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.ineq_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        ineq_vec1 = km.ineq_factory.generate()
        x = PrimalDualVector(p_vec1, ineq_vec=ineq_vec1)
        p_vec2 = km.primal_factory.generate()
        ineq_vec2 = km.ineq_factory.generate()
        r = PrimalDualVector(p_vec2, ineq_vec=ineq_vec2)
        aa.set_initial_data(x)

        # Create synthetic solution and residual data and feed into aa history
        x_scalar = np.linspace(1, max_stored, num=max_stored)
        r_scalar = np.ones_like(x_scalar)
        for k in range(max_stored):
            x.equals(x_scalar[k])
            r.equals(r_scalar[k])
            aa.add_to_history(x, r)
        aa.build_difference_matrices()
        A = np.zeros(15)
        for k in range(max_stored-1):
            aa.x_diff[k].get_base_data(A[:])
            for i in range(A.shape[0]):
                self.assertAlmostEqual(A[i], 1.0, places=14)

    def test_build_difference_matrices_case4(self):
        # case 4: constrained with both equality and inequality constraints
        max_stored = 3
        solver = UserSolver(10, num_eq=3, num_ineq=5)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        ineqf = km.ineq_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant([pf, eqf, ineqf], optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.eq_factory.request_num_vectors(2)
        km.ineq_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        eq_vec1 = km.eq_factory.generate()
        ineq_vec1 = km.ineq_factory.generate()
        x = PrimalDualVector(p_vec1, eq_vec=eq_vec1, ineq_vec=ineq_vec1)
        p_vec2 = km.primal_factory.generate()
        eq_vec2 = km.eq_factory.generate()
        ineq_vec2 = km.ineq_factory.generate()
        r = PrimalDualVector(p_vec2, eq_vec=eq_vec2, ineq_vec=ineq_vec2)
        aa.set_initial_data(x)

        # Create synthetic solution and residual data and feed into aa history
        x_scalar = np.linspace(1, max_stored, num=max_stored)
        r_scalar = np.ones_like(x_scalar)
        for k in range(max_stored):
            x.equals(x_scalar[k])
            r.equals(r_scalar[k])
            aa.add_to_history(x, r)
        aa.build_difference_matrices()
        A = np.zeros(18)
        for k in range(max_stored-1):
            aa.x_diff[k].get_base_data(A[:])
            for i in range(A.shape[0]):
                self.assertAlmostEqual(A[i], 1.0, places=14)

    def test_solve_case1(self):
        # case 1: unconstrained
        max_stored = 4
        solver = UserSolver(3)
        km = KonaMemory(solver)
        pf = km.primal_factory
        optns = {'max_stored': max_stored}
        aa = AndersonMultiSecant(pf, optns)
        # add request that will be used to create x and r
        km.primal_factory.request_num_vectors(2)
        km.allocate_memory()

        # create x and r vectors
        p_vec1 = km.primal_factory.generate()
        x = PrimalDualVector(p_vec1)
        p_vec2 = km.primal_factory.generate()
        r = PrimalDualVector(p_vec2)
        aa.set_initial_data(x)

        # For the following synthetic iterates:
        # Hessian matrix is [1 0 0; 0 100 0; 0 0 10]
        # initial iterate is [1 1 1]
        x.set_base_data(np.array([1.,1.,1.]))
        r.set_base_data(np.array([1.,100.,10.]))
        aa.add_to_history(x, r)
        x.set_base_data(np.array([0.,1.,1.]))
        r.set_base_data(np.array([0.,100.,10.]))
        aa.add_to_history(x, r)
        x.set_base_data(np.array([0.,0.,1.]))
        r.set_base_data(np.array([0.,0.,10.]))
        aa.add_to_history(x, r)
        x.set_base_data(np.array([0.,0.,0.]))
        r.set_base_data(np.array([0.,0.,0.]))
        aa.add_to_history(x, r)
        aa.build_difference_matrices(mu=1.0)

        # AA should be able to recover exact inverse
        x.set_base_data(np.array([1.,0.,0.]))
        aa.solve(x, r)
        A = np.zeros(3)
        r.get_base_data(A[:])
        self.assertAlmostEqual(np.linalg.norm(A - np.array([-1.,0.,0.])), 0.0, places=14)

        x.set_base_data(np.array([0.,100.,0.]))
        aa.solve(x, r)
        r.get_base_data(A[:])
        self.assertAlmostEqual(np.linalg.norm(A - np.array([0.,-1.,0.])), 0.0, places=14)

        x.set_base_data(np.array([0.,0.,10.]))
        aa.solve(x, r)
        r.get_base_data(A[:])
        self.assertAlmostEqual(np.linalg.norm(A - np.array([0.,0.,-1.])), 0.0, places=14)

if __name__ == "__main__":
    unittest.main()
