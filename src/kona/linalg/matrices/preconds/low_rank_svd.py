import numpy as np

from kona.options import get_opt
from kona.linalg.solvers.util import lanczos_bidiag, lanczos_tridiag

class LowRankSVD(object):
    """
    This object produces a low-rank SVD approximation of a matrix defined
    by a given matrix-vector product.

    If there is no reverse matrix-vector product defined, the object
    assumes the matrix to be symmetric and uses a Lanczos tridiagonalization
    algorithm for the decomposition. If a reverse product is defined, then
    Lanczos bi-diagonalization is used for the rectangular matrix.

    Parameters
    ----------
    fwd_mat_vec : function
        Matrix-vector forward product function handle.
    fwd_factory : VectorFactory
        Vectors produced by this factory must be valid input vector for the
        forward product.
    rev_mat_vec : function, optional
        Matrix-vector transpose/reverse product function handle.
    rev_factory : VectorFactory, optional
        Vectors produced by this factory must be valid input vector for the
        transpose/reverse product.
    optns : dict, optional
        Options dictionary.

    Attributes
    ----------
    subspace_size : int
        Number of vectors in each Lanczos subspace. Corresponds to the number
        of iterations in the Lanczos algorithm, and the number of singular
        values approximated by the decomposition.
    """
    def __init__(self, fwd_mat_vec, fwd_factory,
                 rev_mat_vec=None, rev_factory=None,
                 optns={}):

        # set basic internal options
        self.subspace_size = get_opt(optns, 10, 'lanczos_size')

        # get references to individual factories
        self.fwd_mat_vec = fwd_mat_vec
        self.fwd_factory = fwd_factory
        if rev_mat_vec is not None:
            self.rev_mat_vec = rev_mat_vec
            self.rev_factory = rev_factory
        else:
            self.rev_mat_vec = None
            self.rev_factory = self.fwd_factory

        # reset the linearization flag
        self._allocated = False

        # request vector memory for future allocation
        self.fwd_factory.request_num_vectors(2*self.subspace_size + 2)
        self.rev_factory.request_num_vectors(2*self.subspace_size + 1)

    def linearize(self):
        if not self._allocated:
            # this is the first allocation
            # generate subspace vectors
            self.q_work = self.fwd_factory.generate()
            self.p_work = self.rev_factory.generate()
            self.Q = []
            self.V = []
            self.P = []
            self.U = []
            for i in range(self.subspace_size):
                self.Q.append(self.fwd_factory.generate())
                self.V.append(self.fwd_factory.generate())
                self.P.append(self.rev_factory.generate())
                self.U.append(self.rev_factory.generate())
            self.Q.append(self.fwd_factory.generate())
            # flip the allocation flag
            self._allocated = True

        # decompose the matrix
        if self.rev_mat_vec is not None:
            # if rectangular, use bi-diagonalization
            S = lanczos_bidiag(self.fwd_mat_vec, self.Q, self.q_work,
                               self.rev_mat_vec, self.P, self.p_work)
        else:
            # if square, use tri-diagonalization
            S = lanczos_tridiag(self.fwd_mat_vec, self.Q)
            for i in range(len(self.P)):
                self.P[i].equals(self.Q[i])

        # decompose the bi/tri-diagonal matrix
        u_tmp, s_tmp, vT_tmp = np.linalg.svd(S, full_matrices=0)
        v_tmp = vT_tmp.T

        # save the singular values
        self.S = np.diag(s_tmp)

        # calculate V = Q*v_tmp
        for j in range(len(self.V)):
            self.V[j].equals(0.0)
            for i in range(len(v_tmp[:, j])):
                self.q_work.equals(self.Q[i])
                self.q_work.times(v_tmp[i, j])
                self.V[j].plus(self.q_work)

        # calculate U = P*u_tmp
        for j in range(len(self.U)):
            self.U[j].equals(0.0)
            for i in range(len(u_tmp[:, j])):
                self.p_work.equals(self.P[i])
                self.p_work.times(u_tmp[i, j])
                self.U[j].plus(self.p_work)

    def approx_fwd_prod(self, in_vec, out_vec):
        VT_in = np.zeros(len(self.V))
        for i in range(len(self.V)):
            VT_in[i] = self.V[i].inner(in_vec)
        SVT_in = np.dot(self.S, VT_in)
        out_vec.equals(0.0)
        for i in range(len(self.U)):
            out_vec.equals_ax_p_by(1., out_vec, SVT_in[i], self.U[i])

    def approx_rev_prod(self, in_vec, out_vec):
        UT_vec = np.zeros(len(self.U))
        for i in range(len(self.U)):
            UT_vec[i] = self.U[i].inner(in_vec)
        SUT_vec = np.dot(self.S, UT_vec)
        out_vec.equals(0.0)
        for i in range(len(self.V)):
            out_vec.equals_ax_p_by(1., out_vec, SUT_vec[i], self.V[i])

    def inv_schur_prod(self, in_vec, out_vec):
        UT_vec = np.zeros(len(self.U)-1)
        for i in range(len(self.U)-1):
            UT_vec[i] = self.U[i].inner(in_vec)
        S2invUT_vec = np.zeros(len(self.U)-1)
        for i in range(len(self.U)-1):
            S2invUT_vec[i] = UT_vec[i]*(1./self.S[i,i]**2 - 1./self.S[-1,-1]**2)
        out_vec.equals(in_vec)
        out_vec.times(1./self.S[-1,-1]**2)
        for i in range(len(self.U)-1):
            out_vec.equals_ax_p_by(1.0, out_vec, S2invUT_vec[i], self.U[i])
