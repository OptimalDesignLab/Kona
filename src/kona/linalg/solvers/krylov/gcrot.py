import gc
import numpy

from kona.options import get_opt
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import \
    EPS, write_header, write_history, solve_tri, \
    generate_givens, apply_givens, mod_gram_schmidt, mod_GS_normalize

class GCROT(KrylovSolver):
    """
    Generalized Conjugate Residual method with Orthogonalization, Truncated
    """

    def __init__(self, vector_factory, optns={}, dual_factory=None):
        super(GCROT, self).__init__(vector_factory, optns)

        # get relative tolerance
        self.rel_tol = get_opt(optns, 0.5, 'rel_tol')

        # get maximum number of recycled vectors, and set current
        self.max_recycle = get_opt(optns, 10, 'max_recycle')
        self.num_stored = 0
        self.ptr = 0 # index of oldest vector in recycled subspace
        self.max_outer = get_opt(optns, 10, 'max_outer')
        self.max_krylov = get_opt(optns, 50, 'max_matvec')

        # put in memory request
        num_vectors = 2*self.max_iter + 2*self.max_recycle + 4
        self.vec_fac.request_num_vectors(num_vectors)
        self.dual_fac = dual_factory
        if self.dual_fac is not None:
            self.dual_fac.request_num_vectors(2*num_vectors)

        # set empty subpaces
        self.C = []
        self.U = []

    def _generate_vector(self):
        if self.dual_fac is None:
            return self.vec_fac.generate()
        else:
            design = self.vec_fac.generate()
            slack = self.dual_fac.generate()
            primal = CompositePrimalVector(design, slack)
            dual = self.dual_fac.generate()
            return ReducedKKTVector(primal, dual)

    def clear_subspace(self):
        self.num_stored = 0
        self.ptr = 0

        # clear out all the vectors stored in C
        # the data goes back to the stack and is used again later
        for vector in self.C:
            del vector._primal._design
            del vector._primal._slack
            del vector._primal
            del vector._dual
            del vector
        self.C = []

        # clear out all vectors stored in U
        # the data goes back to the stack and is used again later
        for vector in self.U:
            del vector._primal._design
            del vector._primal._slack
            del vector._primal
            del vector._dual
            del vector
        self.U = []

        # force garbage collection
        gc.collect()

    def solve(self, mat_vec, b, x, precond):
        # validate solver options
        self._validate_options()

        # initialize some work data for the outer GCRO method
        C_new = self._generate_vector()
        U_new = self._generate_vector()
        res = self._generate_vector()
        iters = 0

        # calculate and store the initial residual
        mat_vec(x, res)
        res.minus(b)
        res.times(-1.0)
        # find initial guess from recycled subspace
        for k in xrange(self.num_stored):
            alpha = res.inner(self.C[k])
            x.equals_ax_p_by(1.0, x, alpha, self.U[k])
            res.equals_ax_p_by(1.0, res, -alpha, self.C[k])
        beta = res.norm2

        # calculate norm of rhs vector
        norm0 = beta

        if (beta <= self.rel_tol*norm0) or (beta < EPS):
            # system is already solved
            self.out_file.write('GCROT system solved by initial guess.\n')
            return iters, beta

        # output header information
        write_header(self.out_file, 'GCROT', self.rel_tol, beta)
        write_history(self.out_file, 0, beta, norm0)

        # begin outer, GCROT, loop
        ##########################

        for j in xrange(self.max_outer):

            # begin nested FGMRES(fgmres_iter)
            fgmres_iter = self.max_recycle - self.num_stored + self.max_iter

            # initialize some work data for FGMRES
            W = []
            Z = []
            y = numpy.zeros(fgmres_iter + 1)
            sn = numpy.zeros(fgmres_iter + 1)
            cn = numpy.zeros(fgmres_iter + 1)
            H = numpy.zeros((fgmres_iter + 1, fgmres_iter))
            g = numpy.zeros(fgmres_iter+1)
            B = numpy.zeros((self.num_stored, fgmres_iter))

            # normalize residual to get W[0]
            W.append(self._generate_vector())
            W[0].equals(res)
            W[0].divide_by(beta)

            # initialize the RHS of the reduced system
            g[0] = beta
            inner_iters = 0
            lin_depend = False
            for i in xrange(fgmres_iter):

                # check convergence and linear dependence
                if lin_depend and (beta > self.rel_tol*norm0):
                    raise RuntimeError(
                        'GCROT: Arnoldi process breakdown: ' +
                        'H(%i, %i) = %e, however '%(i+1, i, H[i+1, i]) +
                        '||res|| = %e\n'%beta)
                elif beta < self.rel_tol*norm0 or iters >= self.max_krylov:
                    break
                inner_iters += 1
                iters += 1

                # precondition W[i] and store result in Z[i]
                Z.append(self._generate_vector())
                precond(W[i], Z[i])

                # add to krylov subspace
                W.append(self._generate_vector())
                mat_vec(Z[i], W[i+1])

                # orthogonalize W[i+1] against the recycled subspace C[:]
                try:
                    mod_gram_schmidt(i, B, self.C, W[i+1])
                except numpy.linalg.LinAlgError:
                    lin_depend = True

                # now orthonormalize W[i+1] against the W[:i]
                try:
                    mod_GS_normalize(i, H, W)
                except numpy.linalg.LinAlgError:
                    lin_depend = True

                # apply old Givens rotations to new column of the Hessenberg
                # matrix then generate new Givens rotation matrix and apply it
                # to the last two elements of H[i, :] and g
                for k in xrange(i):
                    H[k, i], H[k+1, i] = apply_givens(
                        sn[k], cn[k], H[k, i], H[k+1, i])

                H[i, i], H[i+1, i], sn[i], cn[i] = generate_givens(
                    H[i, i], H[i+1, i])
                g[i], g[i+1] = apply_givens(sn[i], cn[i], g[i], g[i+1])

                # set L2 norm of residual and output relative residual
                beta = abs(g[i+1])
                write_history(self.out_file, iters, beta, norm0)

            # end nested FGMRES
            ###################
            i = inner_iters

            # calculate U_new = (Z - U B)R^{-1} g
            # first, solve to get y = R^{-1} g
            y[:i] = solve_tri(H[:i, :i], g[:i], lower=False)
            U_new.equals(0.0)
            for k in xrange(i):
                U_new.equals_ax_p_by(1.0, U_new, y[k], Z[k])
            # update U_new -= U * B
            for k in xrange(self.num_stored):
                tmp = numpy.dot(B[k, :i], y[:i])
                U_new.equals_ax_p_by(1.0, U_new, -tmp, self.U[k])

            # finished with g, so undo rotations to find C_new
            y[:i] = g[:i]
            y[i] = 0.0
            for k in xrange(i-1, -1, -1):
                y[k], y[k+1] = apply_givens(-sn[k], cn[k], y[k], y[k+1])
            C_new.equals(0.0)
            for k in xrange(i+1):
                C_new.equals_ax_p_by(1.0, C_new, y[k], W[k])

            # normalize and scale new vectors and update solution and res
            alpha = 1.0/C_new.norm2
            C_new.times(alpha)
            U_new.times(alpha)
            alpha = C_new.inner(res)
            res.equals_ax_p_by(1.0, res, -alpha, C_new)
            x.equals_ax_p_by(1.0, x, alpha, U_new)

            # determine which vector to discard from U[:self.num_stored] and
            # C[:self.num_stored], if necessary
            if self.num_stored < self.max_recycle:
                self.ptr = self.num_stored
                self.num_stored += 1
                self.C.append(self._generate_vector())
                self.U.append(self._generate_vector())
            else:
                if self.max_recycle > 0:
                    self.ptr = (self.ptr+1) % self.num_stored
            if self.max_recycle != 0:
                self.C[self.ptr].equals(C_new)
                self.U[self.ptr].equals(U_new)

            # get new residual norm
            # this should be the same as the last iter in FGMRES
            #print 'beta - ||res|| =', beta - res.norm2
            beta = res.norm2

            if beta < self.rel_tol*norm0 or iters >= self.max_krylov:
                break

        # end GCRO loop
        ###############

        if self.check_res:
            # recalculate explicitly and check final residual
            mat_vec(x, res)
            res.equals_ax_p_by(1.0, b, -1.0, res)
            true_res = res.norm2
            self.out_file.write(
                '# GCROT final (true) residual : ' +
                '|res|/|res0| = %e\n'%(true_res/norm0)
            )
            if abs(true_res - beta) > 0.01*self.rel_tol*norm0:
                self.out_file.write(
                    '# WARNING in GCROT: true residual norm and ' +
                    'calculated residual norm do not agree.\n' +
                    '# (res - beta)/res0 = %e\n'%((true_res - beta)/norm0)
                )
            return iters, true_res
        else:
            return iters, beta
