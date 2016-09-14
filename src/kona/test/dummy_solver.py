import numpy

from copy import deepcopy

from kona.user import UserSolver

class DummySolver(UserSolver):

    def __init__(self, num_design=1, num_eq=1, num_ineq=1):
        num_state = num_design
        super(DummySolver, self).__init__(
            num_design, num_state, num_eq, num_ineq)

    def eval_obj(self, at_design, at_state):
        return numpy.sum(at_design) + numpy.sum(at_state.data)

    def eval_residual(self, at_design, at_state, store_here):
        store_here.data[:] = at_design[:] + at_state.data[:]

    def eval_eq_cnstr(self, at_design, at_state):
        return numpy.ones(self.num_eq)*self.eval_obj(at_design, at_state)

    def eval_ineq_cnstr(self, at_design, at_state):
        return numpy.ones(self.num_ineq)*self.eval_obj(at_design, at_state)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = in_vec[:]

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = in_vec.data[:]

    def multiply_dRdX_T(self, at_design, at_state, in_vec):
        return deepcopy(in_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = in_vec.data[:]

    def build_precond(self):
        pass

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = in_vec.data[:]
        return 0

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = in_vec.data[:]
        return 0

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        return numpy.ones(self.num_eq)*numpy.sum(in_vec)

    def multiply_dCEQdU(self, at_design, at_state, in_vec):
        return numpy.ones(self.num_eq)*numpy.sum(in_vec.data)

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        return numpy.ones(self.num_design)*numpy.sum(in_vec)

    def multiply_dCEQdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = numpy.sum(in_vec)

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        return numpy.ones(self.num_ineq)*numpy.sum(in_vec)

    def multiply_dCINdU(self, at_design, at_state, in_vec):
        return numpy.ones(self.num_ineq)*numpy.sum(in_vec.data)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        return numpy.ones(self.num_design)*numpy.sum(in_vec)

    def multiply_dCINdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data[:] = numpy.sum(in_vec)

    def eval_dFdX(self, at_design, at_state):
        return numpy.ones(self.num_design)

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data[:] = 1.0

    def init_design(self):
        return numpy.ones(self.num_design)*10.

    def solve_nonlinear(self, at_design, result):
        result.data[:] = -at_design[:]
        return 1

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = -rhs_vec.data[:]
        return 1

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        result.data[:] = -rhs_vec.data[:]
        return 1
