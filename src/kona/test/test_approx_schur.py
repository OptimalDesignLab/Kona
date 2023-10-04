import numpy as np
import unittest
import inspect
import os

from kona import Optimizer
from kona.algorithms.base_algorithm import OptimizationAlgorithm
from kona.linalg.matrices.preconds.schur import ApproxSchur
from kona.user import UserSolver
from kona.linalg.vectors.composite import PrimalDualVector
# from kona.linalg.matrices.hessian.constraint_jacobian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import TotalConstraintJacobian

class SolverForApproxSchur(UserSolver):
    """A minimal class used to provided synthetic total-constraint Jacobians"""

    def __init__(self, ndv=20, neq=10, nineq=0):
        super(SolverForApproxSchur, self).__init__(
            num_design=ndv,
            num_state=0,
            num_eq=neq,
            num_ineq=nineq)
        self.num_design = ndv
        self.num_eq = neq
        self.num_ineq = nineq
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.dCdX = np.loadtxt(path+'/synthetic_jac.dat')

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        if self.num_eq > 0:
            return np.dot(self.dCdX[0:self.num_eq,:], in_vec)
        else:
            return 0.0

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        if self.num_eq > 0:
            return np.dot(in_vec.T, self.dCdX[0:self.num_eq,:]).T
        else:
            return 0.0

class AlgorithmForApproxSchur(OptimizationAlgorithm,unittest.TestCase):

    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(AlgorithmForApproxSchur, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )
        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(3)
        self.state_factory.request_num_vectors(1)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(2)
        if self.ineq_factory is not None:
            self.eq_factory.request_num_vectors(2)
        max_lanczos = 10
        optns = {'lanczos_size': max_lanczos}
        self.precond = ApproxSchur([primal_factory, state_factory, eq_factory, ineq_factory], optns)
        km = self.primal_factory
        self.num_design = km._memory.ndv
        self.num_eq = km._memory.neq
        self.num_ineq = km._memory.nineq
        # algorithm also needs direct access to the Jacobian
        self.dCdX = TotalConstraintJacobian([primal_factory, state_factory, eq_factory,
                                             ineq_factory])

    def _generate_vector(self):
        dual_eq = None
        dual_ineq = None
        primal = self.primal_factory.generate()
        if self.eq_factory is not None:
            dual_eq = self.eq_factory.generate()
        if self.ineq_factory is not None:
            dual_ineq = self.ineq_factory.generate()
        return PrimalDualVector(primal, eq_vec=dual_eq, ineq_vec=dual_ineq)

    def exact_product(self, in_vec, out_vec):
        """Apply the exact augmented matrix using constraint Jacobian from file"""
        self.dCdX.T.product(in_vec.get_dual(), out_vec.primal)
        out_vec.primal.equals_ax_p_by(1.0, in_vec.primal, -1.0, out_vec.primal)
        self.dCdX.product(in_vec.primal, out_vec.get_dual())
        out_vec.get_dual().times(-1.)

    def solve(self):
        # at_state and at_design are not used, but they need to be defined
        at_state = self.state_factory.generate()
        at_design = self.primal_factory.generate()
        u_vec = self._generate_vector()
        v_vec = self._generate_vector()
        self.dCdX.linearize(at_design, at_state)
        self.precond.linearize(at_design, at_state)
        for i in range(20):
            u_vec.equals(0.)
            v_vec.equals(0.)
            u_vec.primal.base.data[i] = 1.0
            self.exact_product(u_vec, v_vec)
            u_vec.equals(0.)
            self.precond.product(v_vec, u_vec)
            for j in range(20):
                if j == i:
                    self.assertAlmostEqual(u_vec.primal.base.data[j], 1., places=10)
                else:
                    self.assertAlmostEqual(u_vec.primal.base.data[j], 0., places=10)
            for j in range(10):
                self.assertAlmostEqual(u_vec.eq.base.data[j], 0., places=10)

        for i in range(10):
            u_vec.equals(0.)
            v_vec.equals(0.)
            u_vec.eq.base.data[i] = 1.0
            self.exact_product(u_vec, v_vec)
            u_vec.equals(0.)
            self.precond.product(v_vec, u_vec)
            for j in range(20):
                self.assertAlmostEqual(u_vec.primal.base.data[j], 0., places=10)
            for j in range(10):
                if j == i:
                    self.assertAlmostEqual(u_vec.eq.base.data[j], 1., places=10)
                else:
                    self.assertAlmostEqual(u_vec.eq.base.data[j], 0., places=10)

class ApproxSchurTestCase(unittest.TestCase):

    def test_ApproxSchur_eq(self):
        '''test Approximate Schur precondition (equality only)'''
        solver = SolverForApproxSchur(ndv=20, neq=10)
        algorithm = AlgorithmForApproxSchur
        optimizer = Optimizer(solver, algorithm)
        optimizer.solve()

if __name__ == "__main__":
    unittest.main()
