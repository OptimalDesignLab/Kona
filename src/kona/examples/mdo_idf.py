import numpy
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov._fgmres import fgmres as KrylovSolver

from mdo_discipline import MDODiscipline
from kona.user import UserSolverIDF

class ScalableIDF(UserSolverIDF):

    def __init__(self, alpha, mu, nState, initDesign, cout = True):
        # check if number of design variables is even
        if len(initDesign)%2 != 0:
            raise ValueError('ERROR: Odd number of design variables!')
        # store the initial design point
        self.startFrom = initDesign
        # initialize the discipline solver
        self.solver = MDODiscipline(len(self.startFrom)/2, nState, alpha)
        # allocate optimization storage and sizing
        super(ScalableIDF, self).__init__(
            len(self.startFrom), 2*self.solver.nState, 0)
        # store the coupling strength parameters
        self.mu = mu
        # preconditioner call counter for cost measurements
        self.precondCount = 0
        # info print flag
        self.cout = cout

    def splitDesignSpace(self, design):
        realDesign = design[:self.num_real_design]
        [Yu, Yw] = numpy.hsplit(realDesign, 2)
        targState = design[self.num_real_design:]
        [uT, uW] = numpy.hsplit(targState, 2)
        return (Yu, Yw, uT, uW)

    def getResidual(self, design, state):
        # split state vector into discipline components
        Yu, Yw, uT, wT = self.splitDesignSpace(design)
        [u, w] = numpy.hsplit(state, 2)
        # calculate the u component of the residual
        Ru = (1.-self.mu)*self.solver.resKernel(u) \
                - self.mu*self.solver.resKernel(wT) \
                + self.solver.getB(Yu)
        # calculate the w component of the residual
        Rw = self.mu*self.solver.resKernel(uT) \
                +(1.-self.mu)*self.solver.resKernel(w) \
                + self.solver.getB(Yw)
        # stack and return the result
        return numpy.hstack((Ru, Rw))

    def precondWrapper(self, state, vec):
        vecBar = (1.-self.mu)*self.solver.resDerivPrecondProd(state, vec)
        self.precondCount += 1
        return vecBar

    def transPrecondWrapper(self, state, vec):
        vecBar = (1.-self.mu)*self.solver.resDerivTransPrecondProd(state, vec)
        self.precondCount += 1
        return vecBar

    def applyPrecond(self, state, vec):
        # split vectors into discipline components
        [u, w] = numpy.hsplit(state, 2)
        [Vu, Vw] = numpy.hsplit(vec, 2)
        # apply the SSOR preconditioner
        Vu_bar = self.precondWrapper(u, Vu)
        Vw_bar = self.precondWrapper(w, Vw)
        # stack and return the result
        return numpy.hstack((Vu_bar, Vw_bar))

    def applyTransPrecond(self, state, vec):
        # split vectors into discipline components
        [u, w] = numpy.hsplit(state, 2)
        [Vu, Vw] = numpy.hsplit(vec, 2)
        # apply the SSOR preconditioner
        Vu_bar = self.transPrecondWrapper(u, Vu)
        Vw_bar = self.transPrecondWrapper(w, Vw)
        # stack and return the result
        return numpy.hstack((Vu_bar, Vw_bar))

    def nonlinearSolve(self, design):
        # start with an initial guess
        state = numpy.zeros(self.num_state)
        [u, w] = numpy.hsplit(state, 2)
        # calculate initial residual
        R = self.getResidual(design, state)
        [Ru, Rw] = numpy.hsplit(R, 2)
        # print initial residual
        if self.cout:
            print "iter = %i : L2 norm of residual = %e" % \
                (0, numpy.linalg.norm(R))
        # mat-vec products for dR/dState
        dRudu = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda du:(1.-self.mu)*self.solver.dRdStateProd(u, du),
            dtype='float64')
        dRwdw = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda dw:(1.-self.mu)*self.solver.dRdStateProd(w, dw),
            dtype='float64')
        # mat-vec products for the ILU-based preconditioners
        Pu = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:self.precondWrapper(u, v),
            dtype='float64')
        Pw = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:self.precondWrapper(w, v),
            dtype='float64')
        # solution parameters
        max_iter = 100
        res_tol = 1.e-7
        i = 1
        converged = False
        # reset precond count for cost tracking
        self.precondCount = 0
        while i < max_iter:
            # solve linearized system
            dU, infoU = KrylovSolver(dRudu, -Ru, M=Pu)
            dW, infoW = KrylovSolver(dRwdw, -Rw, M=Pw)
            # update guess
            u += dU
            w += dW
            # update residual
            R = self.getResidual(design, numpy.hstack((u, w)))
            [Ru, Rw] = numpy.hsplit(R, 2)
            # print iteration information
            if self.cout:
                print "iter = %i : L2 norm of residual = %e" % \
                    (i, numpy.linalg.norm(R))
            if infoU != 0:
                print "MDO_IDF.nonlinearSolve() >> GMRES for U failed!"
                break
            elif infoW != 0:
                print "MDO_IDF.nonlinearSolve() >> GMRES for W failed!"
                break
            elif (numpy.linalg.norm(Ru) <= res_tol) and \
                 (numpy.linalg.norm(Rw) <= res_tol): # check for convergence
                converged = True
                break
            else:
                i += 1
        # return the solution vector, flags for convergence and solution cost
        state = numpy.hstack((u, w))
        return state, converged, self.precondCount

################################################################################
#####################            KONA FUNCTIONS            #####################
################################################################################

    def eval_obj(self, at_design, at_state):
        # take the state vector data
        state = at_state.data
        cost = 0
        # split the state vector into its discipline components
        [state_u, state_w] = numpy.hsplit(state, 2)
        # calculate the objective function
        obj_val = 0.5*numpy.trapz((state_u**2 + state_w**2), dx=self.solver.dx)
        # return the value and cost of calculation as a tuple
        return (obj_val, cost)

    def eval_residual(self, at_design, at_state, result):
        # check the design and state vectors out of storage
        design = at_design.data
        state = at_state.data
        # store the result
        result.data = self.getResidual(design, state)

    def eval_ceq(self, at_design, at_state, result):
        # check the design and state vectors out of storage
        design = at_design.data
        __, __, uT, uW = self.splitDesignSpace(design)
        targState = numpy.hstack((uT, uW))
        state = at_state.data
        # evaluate IDF constraints and store result
        result.data = targState - state

    def multiply_dRdX(self, at_design, at_state, vec, result):
        # get the design vector
        design = at_design.data
        __, __, uT, wT = self.splitDesignSpace(design)
        # split arbitrary vector
        vec = vec.data
        vecYu, vecYw, vecUt, vecWt = self.splitDesignSpace(vec)
        # calculate design components of the product
        outU = self.solver.dBdDesignProd(vecYu)
        outW = self.solver.dBdDesignProd(vecYw)
        dRdDesignProd = numpy.hstack((outU, outW))
        # calculate target state components of the product
        outU = -self.mu*self.solver.dRdStateProd(wT, vecWt)
        outW = self.mu*self.solver.dRdStateProd(uT, vecUt)
        dRdTargStateProd = numpy.hstack((outU, outW))
        # assemble the complete product and store
        result.data = dRdDesignProd + dRdTargStateProd

    def multiply_dRdU(self, at_design, at_state, vec, result):
        # get the state vector from storage
        state = at_state.data
        [u, w] = numpy.hsplit(state, 2)
        # get the arbitrary vector
        vec = vec.data
        [Vu, Vw] = numpy.hsplit(vec, 2)
        # assemble the result
        outU = (1. - self.mu)*self.solver.dRdStateProd(u, Vu)
        outW = (1. - self.mu)*self.solver.dRdStateProd(w, Vw)
        # store the result
        result.data = numpy.hstack((outU, outW))

    def multiply_dRdX_T(self, at_design, at_state, vec, result):
        # split the design vector
        design = at_design.data
        __, __, uT, wT = self.splitDesignSpace(design)
        # split arbitrary vector
        vec = vec.data
        [vecU, vecW] = numpy.hsplit(vec, 2)
        # calculate design components of the product
        outYu = self.solver.dBdDesignTransProd(vecU)
        outYw = self.solver.dBdDesignTransProd(vecW)
        dRdDesignTransProd = numpy.hstack((outYu, outYw))
        # calculate target state components of the product
        outUt = self.mu*self.solver.dRdStateTransProd(uT, vecW)
        outWt = -self.mu*self.solver.dRdStateTransProd(wT, vecU)
        dRdTargStateTransProd = numpy.hstack((outUt, outWt))
        # assemble the complete product and store
        out = numpy.hstack((dRdDesignTransProd, dRdTargStateTransProd))
        result.data[:] = out[:]

    def multiply_dRdU_T(self, at_design, at_state, vec, result):
        # get the state vector from storage
        state = at_state.data
        [u, w] = numpy.hsplit(state, 2)
        # get the arbitrary vector
        vec = vec.data
        [Vu, Vw] = numpy.hsplit(vec, 2)
        # assemble the result
        outU = (1. - self.mu)*self.solver.dRdStateTransProd(u, Vu)
        outW = (1. - self.mu)*self.solver.dRdStateTransProd(w, Vw)
        # store the result
        result.data = numpy.hstack((outU, outW))

    def apply_precond(self, at_design, at_state, vec, result):
        state = at_state.data
        vec = vec.data
        self.precondCount = 0
        result.data = self.applyPrecond(state, vec)
        return self.precondCount

    def apply_precond_T(self, at_design, at_state, vec, result):
        state = at_state.data
        vec = vec.data
        self.precondCount = 0
        result.data = self.applyTransPrecond(state, vec)
        return self.precondCount

    def multiply_dCdX(self, at_design, at_state, vec, result):
        vec = vec.data
        __, __, vecUt, vecWt = self.splitDesignSpace(vec)
        result.data = numpy.hstack((vecUt, vecWt))

    def multiply_dCdU(self, at_design, at_state, vec, result):
        result.data[:] = -vec.data[:]

    def multiply_dCdX_T(self, at_design, at_state, vec, result):
        dCdyTvec = numpy.zeros(self.num_real_design)
        dCdTargStateTvec = vec.data.copy()
        result.data = numpy.hstack((dCdyTvec, dCdTargStateTvec))

    def multiply_dCdU_T(self, at_design, at_state, vec, result):
        result.data[:] = -vec.data[:]

    def eval_dFdX(self, at_design, at_state, result):
        # for this problem, the gradient of the objective function w.r.t the
        # design variables is zero
        result.data = numpy.zeros(self.num_primal)

    def eval_dFdU(self, at_design, at_state, result):
        # take the state vector out from storage
        v = at_state.data
        # split the state vector into its discipline components
        [u, w] = numpy.hsplit(v, 2)
        # calculate the two components of the gradient
        baseVec = numpy.ones(self.solver.nState)
        dJdu = self.solver.dx*u*baseVec
        dJdu[0] *= 0.5
        dJdu[-1] *= 0.5
        dJdw = self.solver.dx*w*baseVec
        dJdw[0] *= 0.5
        dJdw[-1] *= 0.5
        # merge the components and store the gradient at the specified index
        result.data = numpy.hstack((dJdu, dJdw))

    def init_design(self, store):
        initDesign = numpy.zeros(self.num_primal)
        initDesign[:2*self.solver.nDesign] = self.startFrom
        store.data = initDesign

    def solve_nonlinear(self, at_design, result):
        # get the design vector from storage
        design = at_design.data
        # perform the non-linear solution
        state, converged, cost = self.nonlinearSolve(design)
        # check convergence
        if not converged:
            cost = -cost
        # store result and return solution cost
        result.data = state
        return cost

    def solve_linear(self, at_design, at_state, rhs, tol, result):
        # take the state vectors out of storage
        state = at_state.data
        [u, w] = numpy.hsplit(state, 2)
        # get the RHS vector from storage
        rhs = rhs.data
        [rhsU, rhsW] = numpy.hsplit(rhs, 2)
        rhsU[0] = 0.
        rhsU[-1] = 0.
        rhsW[0] = 0.
        rhsW[-1] = 0.
        # mat-vec products for dR/dState
        dRudu = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:(1.-self.mu)*self.solver.dRdStateProd(u, v),
            dtype='float64')
        dRwdw = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:(1.-self.mu)*self.solver.dRdStateProd(w, v),
            dtype='float64')
        # mat-vec products for the ILU-based preconditioners
        Pu = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:self.precondWrapper(u, v),
            dtype='float64')
        Pw = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:self.precondWrapper(w, v),
            dtype='float64')
        # calculate tolerances
        abs_tol = 1.e-7
        rel_tol = abs_tol/numpy.linalg.norm(rhs)
        # calculate the solution and store them at the specified index
        self.precondCount = 0
        solU, infoU = KrylovSolver(dRudu, rhsU, tol=rel_tol, M=Pu)
        solW, infoW = KrylovSolver(dRwdw, rhsW, tol=rel_tol, M=Pw)
        result.data = numpy.hstack((solU, solW))
        # evaluate FGMRES output for U discipline
        convergedU = False
        if infoU == 0:
            convergedU = True
        elif infoU > 0:
            print "solve_linearsys() >> FGMRES for U: Failed @ %i iter" % info
        else:
            print "solve_linearsys() >> FGMRES for U: Breakdown!"
        # evaluate FGMRES output for W discipline
        convergedW = False
        if infoW == 0:
            convergedW = True
        elif infoW > 0:
            print "solve_linearsys() >> FGMRES for W: Failed @ %i iter" % info
        else:
            print "solve_linearsys() >> FGMRES for W: Breakdown!"
        # check convergence and return cost
        if convergedU and convergedW:
            return self.precondCount
        else:
            return -self.precondCount

    def solve_adjoint(self, at_design, at_state, rhs, tol, result):
        # take the state vectors out of storage
        state = at_state.data
        [u, w] = numpy.hsplit(state, 2)
        # if rhs index is negative, use -dJ/dState as the RHS vector
        if rhs < 0:
            # calculate the two components of the gradient
            baseVec = numpy.ones(self.solver.nState)
            rhsU = -self.solver.dx*u*baseVec
            rhsU[0] *= 0.5
            rhsU[-1] *= 0.5
            rhsW = -self.solver.dx*w*baseVec
            rhsW[0] *= 0.5
            rhsW[-1] *= 0.5
        # otherwise use whatever rhs index is provided
        else:
            rhs = rhs.data
            [rhsU, rhsW] = numpy.hsplit(rhs, 2)
            rhsU[0] = 0.
            rhsU[-1] = 0.
            rhsW[0] = 0.
            rhsW[-1] = 0.
        # mat-vec products for dR/dState
        dRuduT = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:(1.-self.mu)*self.solver.dRdStateTransProd(u, v),
            dtype='float64')
        dRwdwT = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:(1.-self.mu)*self.solver.dRdStateTransProd(w, v),
            dtype='float64')
        # mat-vec products for the ILU-based preconditioners
        PTu = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:self.transPrecondWrapper(u, v),
            dtype='float64')
        PTw = LinearOperator((self.solver.nState, self.solver.nState),
            matvec= lambda v:self.transPrecondWrapper(w, v),
            dtype='float64')
        # calculate tolerances
        abs_tol = 1.e-7
        rel_tol = abs_tol/numpy.linalg.norm(rhs)
        # calculate the solution and store them at the specified index
        self.precondCount = 0
        solU, infoU = KrylovSolver(dRuduT, rhsU, tol=rel_tol, M=PTu)
        solW, infoW = KrylovSolver(dRwdwT, rhsW, tol=rel_tol, M=PTw)
        result.data = numpy.hstack((solU, solW))
        # evaluate FGMRES output for U discipline
        convergedU = False
        if infoU == 0:
            convergedU = True
        elif infoU > 0:
            print "solve_linearsys() >> FGMRES for U: Failed @ %i iter" % info
        else:
            print "solve_linearsys() >> FGMRES for U: Breakdown!"
        # evaluate FGMRES output for W discipline
        convergedW = False
        if infoW == 0:
            convergedW = True
        elif infoW > 0:
            print "solve_linearsys() >> FGMRES for W: Failed @ %i iter" % info
        else:
            print "solve_linearsys() >> FGMRES for W: Breakdown!"
        # check convergence and return cost
        if convergedU and convergedW:
            return self.precondCount
        else:
            return -self.precondCount

    def current_solution(self, curr_design, curr_state, curr_adj, curr_dual, num_iter):
        self.current_design = curr_design.data
        Yu, Yw, __, __ = self.splitDesignSpace(self.current_design)
        self.current_state = curr_state.data
        if self.cout:
            print numpy.hstack((Yu, Yw))
