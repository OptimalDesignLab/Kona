import unittest

import numpy as np

from kona.linalg.memory import KonaMemory
from kona.examples import Spiral, Simple2x2
from kona.linalg.matrices.hessian import ReducedHessian
from kona.options import BadKonaOption


class ReducedHessianTestCase(unittest.TestCase):
    '''Test case for the Reduced Hessian approximation matrix.'''

    def setUp(self):
        #solver = Simple2x2()
        solver = Spiral()
        km = KonaMemory(solver)
        self.pf = km.primal_factory
        self.sf = km.state_factory

        self.pf.request_num_vectors(5)
        self.sf.request_num_vectors(3)

        self.hessian = ReducedHessian([self.pf, self.sf])

        km.allocate_memory()

    def assertRelError(self, vec1, vec2, atol=1e-15):
        self.assertTrue(np.linalg.norm(vec1 - vec2) < atol)

    def test_product(self):
        # get memory
        x = self.pf.generate()
        dJdX = self.pf.generate()
        dJdX_pert = self.pf.generate()
        primal_work = self.pf.generate()
        v = self.pf.generate()
        state = self.sf.generate()
        adjoint = self.sf.generate()
        state_work = self.sf.generate()

        # calculate total derivative at current design
        init_design = 3*np.pi
        #init_design = 1.0
        x.equals(init_design)
        state.equals_primal_solution(x)
        adjoint.equals_adjoint_solution(x, state, state_work)
        dJdX.equals_total_gradient(x, state, adjoint, primal_work)
        # calculate total derivative at perturbed bdesign
        epsilon_fd = 1e-5
        v.equals(2.0)
        x.equals_ax_p_by(1.0, x, epsilon_fd, v)
        state.equals_primal_solution(x)
        adjoint.equals_adjoint_solution(x, state, state_work)
        dJdX_pert.equals_total_gradient(x, state, adjoint, primal_work)
        # calculate directional derivative of the total derivative
        # this is the FD approximation of the Hessian-vector product
        dJdX_pert.minus(dJdX)
        dJdX_pert.divide_by(epsilon_fd)

        # reset the design point and linearize the Hessian
        x.equals(init_design)
        state.equals_primal_solution(x)
        adjoint.equals_adjoint_solution(x, state, state_work)
        self.hessian.linearize(x, state, adjoint)

        # perform the hessian-vector product
        self.hessian.product(v, dJdX)

        primal_work.equals_ax_p_by(1.0, dJdX, -1.0, dJdX_pert)
        diff_norm = primal_work.norm2

        self.assertTrue(diff_norm <= 1e-5*dJdX.norm2)


if __name__ == "__main__":
    unittest.main()
