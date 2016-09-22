import unittest

from kona.examples import Sellar
from kona.linalg.memory import KonaMemory
from kona.linalg.matrices.common import dCdU, dRdU
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector


class TotalConstraintJacobianTestCase(unittest.TestCase):
    '''Test case for the total constraint jacobian approximation matrix.'''

    def _generate_KKT_vector(self):
        design = self.pf.generate()
        slack = self.df.generate()
        primal = CompositePrimalVector(design, slack)
        dual = self.df.generate()
        return ReducedKKTVector(primal, dual)

    def test_constrained_product(self):
        solver = Sellar()
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory
        self.df = km.ineq_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(15)

        self.A = TotalConstraintJacobian([self.pf, self.sf, self.df])

        km.allocate_memory()

        # get memory
        primal_work = self.pf.generate()
        state = self.sf.generate()
        adjoint = self.sf.generate()
        state_work = self.sf.generate()
        dual_work = self.df.generate()
        X = self._generate_KKT_vector()
        dLdX = self._generate_KKT_vector()
        dLdX_pert = self._generate_KKT_vector()
        in_vec = self._generate_KKT_vector()
        out_vec = self._generate_KKT_vector()

        in_vec.equals(2.0)

        X.equals_init_guess()
        state.equals_primal_solution(X.primal.design)
        state_work.equals_objective_partial(X.primal.design, state)
        dCdU(X.primal.design, state).T.product(X.dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X.primal.design, state).T.solve(state_work, adjoint)
        dLdX.equals_KKT_conditions(X, state, adjoint, 0.0)

        epsilon_fd = 1e-6
        X.primal.design.equals_ax_p_by(
            1.0, X.primal.design, epsilon_fd, in_vec.primal.design)
        state.equals_primal_solution(X.primal.design)
        state_work.equals_objective_partial(X.primal.design, state)
        dCdU(X.primal.design, state).T.product(X.dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X.primal.design, state).T.solve(state_work, adjoint)
        dLdX_pert.equals_KKT_conditions(X, state, adjoint, 0.0)

        dLdX_pert.minus(dLdX)
        dLdX_pert.divide_by(epsilon_fd)

        X.equals_init_guess()
        state.equals_primal_solution(X.primal.design)
        state_work.equals_objective_partial(X.primal.design, state)
        dCdU(X.primal.design, state).T.product(X.dual, adjoint)
        state_work.plus(adjoint)
        state_work.times(-1.)
        dRdU(X.primal.design, state).T.solve(state_work, adjoint)
        self.A.linearize(X.primal.design, state)
        self.A.product(in_vec.primal.design, out_vec.dual)

        print '-----------------------------'
        print 'Constraint Hessian'
        print '-----------------------------'
        print 'FD product:'
        print dLdX_pert.dual.base.data
        print 'Analytical product:'
        print out_vec.dual.base.data
        print '-----------------------------'

        dLdX.equals_ax_p_by(1.0, dLdX_pert, -1.0, out_vec)
        diff_norm = dLdX.dual.norm2

        self.assertTrue(diff_norm <= 1e-3)

if __name__ == "__main__":
    unittest.main()
