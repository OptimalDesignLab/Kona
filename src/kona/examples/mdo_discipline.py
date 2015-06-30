import numpy
from scipy import sparse
from scipy.sparse.linalg import spilu

class MDODiscipline(object):
    """
    A base class for a nonlinear PDE governed MDO discipline. Both MDF and
    IDF architectures can be derived using this object.
    """

    def __init__(self, nDesign, nState, alpha):
        # store the non-linearity coefficients
        self.alpha = alpha
        # store the number of design and state variables
        self.nDesign = nDesign
        self.nState = nState
        # calculate the discretization step size
        self.dx = 1./(self.nState-1)
        # prepare basic arays for diagonal matrices
        diag = numpy.ones(self.nState-2)
        off = numpy.ones(self.nState-3)
        triDiag = numpy.array([off, diag, off])
        stencil = [-1, 0, 1]
        # store the sparse B[1, -2, 1] banded matrix
        coeffs = numpy.array([1., -2., 1.])
        data = coeffs*triDiag
        self.B121 = sparse.diags(data, stencil, format="lil")
        # store the sparse B[0, -1, 1] banded matrix
        coeffs = numpy.array([0., -1., 1.])
        data = coeffs*triDiag
        self.B011_neg = sparse.diags(data, stencil, format="lil")
        # store the sparse B[-1, 1, 0] banded matrix
        coeffs = numpy.array([-1., 1., 0.])
        data = coeffs*triDiag
        self.B110_neg = sparse.diags(data, stencil, format="lil")
        # store the sparse B[0, 1, 1] banded matrix
        coeffs = numpy.array([0., 1., 1.])
        data = coeffs*triDiag
        self.B011 = sparse.diags(data, stencil, format="lil")
        # store the sparse B[1, 1, 0] banded matrix
        coeffs = numpy.array([1., 1., 0.])
        data = coeffs*triDiag
        self.B110 = sparse.diags(data, stencil, format="lil")

    def resKernel(self, state):
        # calculate the first residual term
        term1 = (self.alpha/(self.dx**2))*self.B121.dot(state[1:-1])
        # modify the B[0,-1,1] and B[-1,1,0] banded matrices
        term3 = self.B011_neg.dot(state[1:-1])
        term4 = self.B110_neg.dot(state[1:-1])
        for i in range(1, self.nState-1):
            term3[i-1] *= (state[i] + state[i+1])**2
            term4[i-1] *= (state[i-1] + state[i])**2
        # use the modified matrices to calculate the 2nd residual term
        term2 = (self.alpha/(4.*self.dx))*(term3 - term4)
        # assemble the residual vector and return it
        res = numpy.zeros(self.nState)
        res[1:-1] = (term1 + term2)
        return res

    def resDerivKernel(self, state):
        # calculate the first jacobian term
        term1 = (self.alpha/(self.dx**2))*self.B121
        # modify other banded matrices with nossn-linear terms
        term3 = self.B011_neg
        term4 = self.B011
        term5 = self.B110_neg
        term6 = self.B110
        for i in range(1, self.nState-1):
            term3[i-1, :] *= (state[i] + state[i+1])**2.
            term4[i-1, :] *= 2.*(state[i] + state[i+1])*(-state[i] + state[i+1])
            term5[i-1, :] *= (state[i-1] + state[i])**2.
            term6[i-1, :] *= 2.*(state[i-1] + state[i])*(-state[i-1] + state[i])
        # use the modified banded matrices to assemble the second jacobian term
        term2 = (self.alpha/(4.*self.dx))*(term3 + term4 - term5 - term6)
        # assemble the final jacobian and return it
        return (term1 + term2)

    def dRdStateProd(self, state, vec):
        # get the state jacobian of the residual
        dRdState = self.resDerivKernel(state)
        # return the jacobian-vector product
        result = numpy.zeros(self.nState)
        result[1:-1] = dRdState.dot(vec[1:-1])
        return result

    def dRdStateTransProd(self, state, vec):
        # get the state jacobian of the residual
        dRdState = self.resDerivKernel(state)
        # return the jacobian-vector product
        result = numpy.zeros(self.nState)
        result[1:-1] = dRdState.transpose().dot(vec[1:-1])
        return result

    def getB(self, design):
        b = numpy.zeros(self.nState)
        for i in range(self.nState):
            for j in range(self.nDesign):
                b[i] += design[j]*numpy.sin(i*(j+1)*numpy.pi/(self.nState - 1.))
        return b

    def dBdDesignProd(self, vec):
        result = numpy.zeros(self.nState)
        for i in range(self.nState):
            for j in range(self.nDesign):
                result[i] += vec[j]*numpy.sin(i*(j+1)*numpy.pi/(self.nState - 1.))
        return result

    def dBdDesignTransProd(self, vec):
        result = numpy.zeros(self.nDesign)
        for j in range(self.nDesign):
            for i in range(self.nState):
                result[j] += vec[i]*numpy.sin(i*(j+1)*numpy.pi/(self.nState - 1.))
        return result

    def resDerivPrecondProd(self, state, vec):
        dRdState = self.resDerivKernel(state)
        P = spilu(sparse.csc_matrix(dRdState))
        vecBar = numpy.zeros(self.nState)
        vecBar[1:-1] = P.solve(vec[1:-1])
        return vecBar

    def resDerivTransPrecondProd(self, state, vec):
        dRdState = self.resDerivKernel(state)
        PT = spilu(sparse.csc_matrix(dRdState.transpose()))
        vecBar = numpy.zeros(self.nState)
        vecBar[1:-1] = PT.solve(vec[1:-1])
        return vecBar
