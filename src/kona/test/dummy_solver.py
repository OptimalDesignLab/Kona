from kona.user import UserSolver
from kona.user import BaseVector

class DummySolver(UserSolver):

    def eval_obj(self, at_design, at_state):
        if isinstance(at_design, BaseVector) and isinstance(at_state, BaseVector):
            return 1.
        else:
            return 0.

    def eval_residual(self, at_design, at_state, result):
        pass

    def eval_ceq(self, at_design, at_state, result):
        pass

    def multiply_jac_d(self, at_design, at_state, in_vec, out_vec):
