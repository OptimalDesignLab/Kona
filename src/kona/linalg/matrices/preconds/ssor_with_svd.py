import numpy as np

from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.preconds import LowRankSVD

class SSORwithSVD(LowRankSVD):
    def __init__(self, vector_factories, optns={}):
        super(SSORwithSVD, self).__init__(vector_factories, optns)

        # request vector memory for future allocation
        self.primal_factory.request_num_vectors(1)
        self.dual_factory.request_num_vectors(1)

        # child class allocation flag
        self._ssor_allocated = False

    def linearize(self, at_kkt, at_state, at_adjoint):
        super(SSORwithSVD, self).linearize(at_kkt, at_state, at_adjoint)

        if not self._ssor_allocated:
            self.design_save = self.primal_factory.generate()
            self.dual_save = self.dual_factory.generate()
            self._ssor_allocated = True

    def mult_sigma_lambda(self, in_vec, out_vec):
        out_vec.equals(in_vec)
        out_vec.times(self.at_dual)
        out_vec.times(self.slack_term)

    def mult_lambdaA_approx(self, in_vec, out_vec):
        VT_in = np.zeros(len(self.V))
        for i in xrange(len(self.V)):
            VT_in[i] = self.V[i].inner(in_vec)
        SVT_in = np.dot(self.S, VT_in)
        out_vec.equals(0.0)
        for i in xrange(len(self.U)):
            self.p_work.equals(self.U[i])
            self.p_work.times(SVT_in[i])
            out_vec.plus(self.p_work)

    def mult_ATlambda_approx(self, in_vec, out_vec):
        UT_in = np.zeros(len(self.U))
        for i in xrange(len(self.U)):
            UT_in[i] = self.U[i].inner(in_vec)
        SUT_in = np.dot(self.S, UT_in)
        out_vec.equals(0.0)
        for i in xrange(len(self.V)):
            self.q_work.equals(self.V[i])
            self.q_work.times(SUT_in[i])
            out_vec.plus(self.q_work)

    def product(self, in_vec, out_vec):
        # do some aliasing
        if self.at_slack is not None:
            in_design = in_vec._primal._design
            in_slack = in_vec._primal._slack
            in_slack.restrict()
            out_design = out_vec._primal._design
            out_slack = out_vec._primal._slack
        else:
            in_design = in_vec._primal
            in_slack = None
            out_design = out_vec._primal
            out_slack = None
        in_dual = in_vec._dual
        out_dual = out_vec._dual

        # create an identity preconditioner for nested solves
        eye = IdentityMatrix()
        precond = eye.product

        # define some SSOR loop parameters
        max_iter = 20
        iters = 0
        step_tol = 0.1

        # set the initial trial point
        self.design_save.equals(0.0)
        self.dual_save.equals(0.0)

        # start SSOR loops
        while iters < max_iter:
            iters += 1

            # assemble the RHS for the design solve
            self.mult_ATlambda_approx(self.dual_save, self.design_work)
            self.design_work.times(-1.)
            self.design_work.plus(in_design)

            # solve the Hessian block
            self.krylov_design.solve(
                self.mult_W_approx, self.design_work, out_design, precond)

            # assemble the RHS for the dual solve
            self.mult_lambdaA_approx(out_design, self.dual_work)
            self.dual_work.plus(in_slack)
            self.dual_work.times(-1.)
            self.p_work.equals(in_dual)
            self.p_work.times(self.at_dual)
            self.dual_work.plus(self.p_work)

            # solve the dual block
            self.krylov_dual.solve(
                self.mult_sigma_lambda, self.dual_work, out_dual, precond)

            # rebuild the design RHS with the new dual solution
            self.mult_ATlambda_approx(out_dual, self.design_work)
            self.design_work.times(-1.)
            self.design_work.plus(in_design)

            # solve the Hessian block again
            self.krylov_design.solve(
                self.mult_W_approx, self.design_work, out_design, precond)

            # check convergence
            self.design_work.equals_ax_p_by(
                1.0, out_design, -1.0, self.design_save)
            design_norm = self.design_work.norm2
            self.dual_work.equals_ax_p_by(
                1.0, out_dual, -1.0, self.dual_save)
            dual_norm = self.dual_work.norm2
            if (design_norm <= step_tol) and (dual_norm <= step_tol):
                break
            else:
                self.design_save.equals(out_design)
                self.dual_save.equals(out_dual)

        # assemble the RHS vector for the slack system
        self.dual_work.equals(out_dual)
        self.dual_work.times(self.at_dual)
        self.dual_work.times(self.slack_term)
        self.dual_work.plus(in_slack)
        self.dual_work.times(-1.)
        self.dual_work.restrict()

        # back out the slack solution
        self.krylov_dual.solve(
            self.mult_sigma_lambda, self.dual_work, out_slack, precond)

        # finally recover the actual dual solution
        out_dual.times(self.at_dual)
