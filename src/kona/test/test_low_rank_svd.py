import unittest

import inspect, os
import numpy as np

from kona.examples import Sellar
from kona.user import UserSolver
from kona.linalg.memory import KonaMemory
from kona.linalg.matrices.common import dCdU, dRdU
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import LagrangianHessian
from kona.linalg.matrices.preconds import LowRankSVD
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositeFactory, CompositePrimalVector, \
    CompositeDualVector


class LowRankSVDTestCase(unittest.TestCase):

    def _generate_KKT_vector(self):
        design = self.pf.generate()
        slack = self.df.generate()
        primal = CompositePrimalVector(design, slack)
        dual = self.df.generate()
        return ReducedKKTVector(primal, dual)

    def test_with_rectangular(self):
        '''LowRankSVD approximation for rectangular matrix'''
        solver = Sellar()
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.ineq_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(15)

        self.A = TotalConstraintJacobian([self.pf, self.sf, self.df])

        def fwd_mat_vec(in_vec, out_vec):
            self.A.product(in_vec, out_vec)

        def rev_mat_vec(in_vec, out_vec):
            self.A.T.product(in_vec, out_vec)

        svd_optns = {'lanczos_size': 3}
        self.svd = LowRankSVD(
            fwd_mat_vec, self.pf, rev_mat_vec, self.df, svd_optns)

        km.allocate_memory()

        X = self._generate_KKT_vector()
        in_vec = self._generate_KKT_vector()
        out_vec_exact = self._generate_KKT_vector()
        out_vec_approx = self._generate_KKT_vector()
        state = self.sf.generate()

        in_vec.equals(2.0)
        X.equals_init_guess()
        state.equals_primal_solution(X.primal.design)
        self.A.linearize(X.primal.design, state)
        self.svd.linearize()

        self.A.product(in_vec.primal.design, out_vec_exact.dual)
        self.svd.approx_fwd_prod(in_vec.primal.design, out_vec_approx.dual)

        print 'Constraint Jacobian test:'

        print 'Exact fwd product  =', out_vec_exact.dual.norm2
        print 'Approx fwd product =', out_vec_approx.dual.norm2

        out_vec_approx.dual.minus(out_vec_exact.dual)
        abs_error = out_vec_approx.dual.norm2
        rel_error = abs_error/out_vec_exact.dual.norm2

        print 'Abs error norm     =', abs_error
        print 'Rel error norm     =', rel_error

        self.assertTrue(rel_error <= 1e-8)

        self.A.T.product(in_vec.dual, out_vec_exact.primal.design)
        self.svd.approx_rev_prod(in_vec.dual, out_vec_approx.primal.design)

        print 'Exact rev product  =', out_vec_exact.primal.design.norm2
        print 'Approx fwd product =', out_vec_approx.primal.design.norm2

        out_vec_approx.primal.design.minus(out_vec_exact.primal.design)
        abs_error = out_vec_approx.primal.design.norm2
        rel_error = abs_error/out_vec_exact.primal.design.norm2

        print 'Abs error norm     =', abs_error
        print 'Rel error norm     =', rel_error

        self.assertTrue(rel_error <= 1e-8)

    def test_with_square(self):
        '''LowRankSVD approximation for square matrix'''
        solver = Sellar()
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.ineq_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(15)

        self.W = LagrangianHessian([self.pf, self.sf, self.df])

        def mat_vec(in_vec, out_vec):
            self.W.multiply_W(in_vec, out_vec)

        svd_optns = {'lanczos_size': 3}
        self.svd = LowRankSVD(
            mat_vec, self.pf, optns=svd_optns)

        km.allocate_memory()

        X = self._generate_KKT_vector()
        in_vec = self._generate_KKT_vector()
        out_vec_exact = self._generate_KKT_vector()
        out_vec_approx = self._generate_KKT_vector()
        state = self.sf.generate()
        state_work = self.sf.generate()
        adjoint = self.sf.generate()

        in_vec.equals(2.0)
        X.equals_init_guess()
        state.equals_primal_solution(X.primal.design)
        state_work.equals_objective_partial(X.primal.design, state)
        dCdU(X.primal.design, state).T.product(X.dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X.primal.design, state).T.solve(state_work, adjoint)
        self.W.linearize(X, state, adjoint)
        self.svd.linearize()

        self.W.multiply_W(in_vec.primal.design, out_vec_exact.primal.design)
        self.svd.approx_fwd_prod(
            in_vec.primal.design, out_vec_approx.primal.design)

        print 'Hessian test:'
        print 'Exact product  =', out_vec_exact.primal.design.norm2
        print 'Approx product =', out_vec_approx.primal.design.norm2

        out_vec_approx.primal.design.minus(out_vec_exact.primal.design)
        abs_error = out_vec_approx.primal.design.norm2
        rel_error = abs_error/out_vec_exact.primal.design.norm2

        print 'Abs error norm =', abs_error
        print 'Rel error norm =', rel_error

        self.assertTrue(rel_error <= 1e-8)

    def test_inv_schur_prod_eq(self):
        '''LowRankSVD approximation of inverse of Schur complement (equality only)'''
        max_lanczos = 10
        solver = UserSolver(20, num_eq=10)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory

        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        A = np.loadtxt(path+'/synthetic_jac.dat')
        Sinv = np.loadtxt(path+'/synthetic_schur.dat')

        # define the Low-rank SVD
        def fwd_mat_vec(in_vec, out_vec):
            out_vec.base.data[:] = A.dot(in_vec.base.data)

        def rev_mat_vec(in_vec, out_vec):
            out_vec.base.data[:] = A.T.dot(in_vec.base.data)

        svd_optns = {'lanczos_size': max_lanczos}
        self.svd = LowRankSVD(
            fwd_mat_vec, pf, rev_mat_vec, eqf, svd_optns)

        # add request that will be used to create in_vec and out_vec
        km.eq_factory.request_num_vectors(2)
        km.allocate_memory()

        in_vec = km.eq_factory.generate()
        out_vec = km.eq_factory.generate()

        # loop over and check each column in the approximate Schur
        self.svd.linearize()
        for i in xrange(km.neq):
            in_vec.base.data = np.zeros_like(in_vec.base.data)
            in_vec.base.data[i] = 1.0
            out_vec.base.data = np.zeros_like(out_vec.base.data)
            self.svd.inv_schur_prod(in_vec, out_vec)
            for j in xrange(km.neq):
                self.assertAlmostEqual(Sinv[j,i]/out_vec.base.data[j], 1.0, places=9)

    def test_inv_schur_prod_ineq(self):
        '''LowRankSVD approximation of inverse of Schur complement (inequality only)'''
        max_lanczos = 10
        solver = UserSolver(20, num_ineq=10)
        km = KonaMemory(solver)
        pf = km.primal_factory
        ineqf = km.ineq_factory

        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        A = np.loadtxt(path+'/synthetic_jac.dat')
        Sinv = np.loadtxt(path+'/synthetic_schur.dat')

        # define the Low-rank SVD
        def fwd_mat_vec(in_vec, out_vec):
            out_vec.base.data[:] = A.dot(in_vec.base.data)

        def rev_mat_vec(in_vec, out_vec):
            out_vec.base.data[:] = A.T.dot(in_vec.base.data)

        svd_optns = {'lanczos_size': max_lanczos}
        self.svd = LowRankSVD(
            fwd_mat_vec, pf, rev_mat_vec, ineqf, svd_optns)

        # add request that will be used to create in_vec and out_vec
        km.ineq_factory.request_num_vectors(2)
        km.allocate_memory()

        in_vec = km.ineq_factory.generate()
        out_vec = km.ineq_factory.generate()

        # loop over and check each column in the approximate Schur
        self.svd.linearize()
        for i in xrange(km.nineq):
            in_vec.base.data = np.zeros_like(in_vec.base.data)
            in_vec.base.data[i] = 1.0
            out_vec.base.data = np.zeros_like(out_vec.base.data)
            self.svd.inv_schur_prod(in_vec, out_vec)
            for j in xrange(km.nineq):
                self.assertAlmostEqual(Sinv[j,i]/out_vec.base.data[j], 1.0, places=9)

    def test_inv_schur_prod_comp(self):
        '''LowRankSVD approximation of inverse of Schur complement (CompositeDualVector)'''
        max_lanczos = 10
        solver = UserSolver(20, num_eq=6, num_ineq=4)
        km = KonaMemory(solver)
        pf = km.primal_factory
        eqf = km.eq_factory
        ineqf = km.ineq_factory
        dualf = CompositeFactory(km, CompositeDualVector)

        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        A = np.loadtxt(path+'/synthetic_jac.dat')
        Sinv = np.loadtxt(path+'/synthetic_schur.dat')

        # define the Low-rank SVD
        def fwd_mat_vec(in_vec, out_vec):
            out_copy = np.zeros(10)
            out_copy[:] = A.dot(in_vec.base.data)
            out_vec.eq.base.data[:] = out_copy[0:6]
            out_vec.ineq.base.data[:] = out_copy[6:]

        def rev_mat_vec(in_vec, out_vec):
            in_copy = np.zeros(10)
            in_copy[0:6] = in_vec.eq.base.data[:]
            in_copy[6:] = in_vec.ineq.base.data[:]
            out_vec.base.data[:] = A.T.dot(in_copy)

        svd_optns = {'lanczos_size': max_lanczos}
        self.svd = LowRankSVD(
            fwd_mat_vec, pf, rev_mat_vec, dualf, svd_optns)

        # add request that will be used to create in_vec and out_vec
        dualf.request_num_vectors(2)
        km.allocate_memory()

        in_vec = dualf.generate()
        out_vec = dualf.generate()

        # loop over and check each column in the approximate Schur
        self.svd.linearize()
        for i in xrange(km.neq):
            in_vec.equals(0.)
            in_vec.eq.base.data[i] = 1.0
            out_vec.equals(0.)
            self.svd.inv_schur_prod(in_vec, out_vec)
            for j in xrange(km.neq):
                self.assertAlmostEqual(Sinv[j,i]/out_vec.eq.base.data[j], 1.0, places=9)
            for j in xrange(km.nineq):
                self.assertAlmostEqual(Sinv[km.neq+j,i]/out_vec.ineq.base.data[j], 1.0, places=9)
        for i in xrange(km.nineq):
            in_vec.equals(0.)
            in_vec.ineq.base.data[i] = 1.0
            out_vec.equals(0.)
            self.svd.inv_schur_prod(in_vec, out_vec)
            for j in xrange(km.neq):
                self.assertAlmostEqual(Sinv[j,km.neq+i]/out_vec.eq.base.data[j], 1.0, places=9)
            for j in xrange(km.nineq):
                self.assertAlmostEqual(Sinv[km.neq+j,km.neq+i]/out_vec.ineq.base.data[j], 1.0,
                                       places=9)

if __name__ == "__main__":
    unittest.main()
