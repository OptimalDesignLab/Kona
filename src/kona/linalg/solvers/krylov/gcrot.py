import numpy

from kona.options import get_opt
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import \
    EPS, write_header, write_history, \
    generate_givens, apply_givens, mod_gram_schmidt

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

        # put in memory request
        self.vec_fac.request_num_vectors(2*self.max_iter + 2*self.max_recycle + 3)
        self.dual_fac = dual_factory
        if self.dual_fac is not None:
            self.dual_fac.request_num_vectors(4*self.max_iter + 4*self.max_recycle + 6)

    def _generate_vector(self):
        if self.dual_fac is None:
            return self.vec_fac.generate()
        else:
            design = self.vec_fac.generate()
            slack = self.dual_fac.generate()
            primal = CompositePrimalVector(design, slack)
            dual = self.dual_fac.generate()
            return ReducedKKTVector(primal, dual)

    def clear_subpace(self):
        self.num_stored = 0
        self.ptr = 0

        # ALP!!!! please check this
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
        Cnew = self._generate_vector()
        Unew = self._generate_vector()
        res = self._generate_vector()
        iters = 0

        # calculate norm of rhs vector
        norm0 = b.norm2

        # calculate and store the initial residual
        mat_vec(x, res)
        res.minus(b)
        res.times(-1.0) # ALP: I am assuming the residual is r = b - Ax
        beta = res.norm2

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
            y = numpy.zeros(fgmres_iter)
            sn = numpy.zeros(fgmres_iter + 1)
            cn = numpy.zeros(fgmres_iter + 1)
            H = numpy.matrix(numpy.zeros((fgmres_iter + 1, fgmres_iter)))
            g = numpy.zeros(fgmres_iter+1)
            B = numpy.matrix(self.num_stored, self.num_stored)

            # normalize residual to get W[0]
            W.append(self._generate_vector())
            W[0].equals(res)
            W[0].divide_by(beta)

            # initialize the RHS of the reduced system
            g[0] = beta

            lin_depend = False
            for i in xrange(fgmres_iter):

                # check convergence and linear dependence
                if lin_depend and (beta > self.rel_tol*norm0):
                    raise RuntimeError(
                        'GCROT: Arnoldi process breakdown: ' +
                        'H(%i, %i) = %e, however '%(i+1, i, H[i+1, i]) +
                        '||res|| = %e\n'%beta)
                elif beta < self.rel_tol*norm0:
                    break

                iters += 1

                # precondition W[i] and store result in Z[i]
                Z.append(self._generate_vector())
                precond(W[i], Z[i])

                # add to krylov subspace
                W.append(self._generate_vector())
                mat_vec(Z[i], W[i+1])

                # orthogonalize W[i+1] against the recycled subspace C[:]
                try:
                    mod_gram_schmidt(i, B, C, W[i+1])
                except numpy.linalg.LinAlgError:
                    self.lin_depend = True

                # now orthogonalize W[i+1] against the W[:i]
                try:
                    mod_gram_schmidt(i, H, W)
                except numpy.linalg.LinAlgError:
                    self.lin_depend = True

                # apply old Givens rotations to new column of the Hessenberg
                # matrix then generate new Givens rotation matrix and apply it
                # to the last two elements of H[i, :] and g
                for k in xrange(i):
                    H[k, i], H[k+1, i] = apply_givens(
                    sn[k], cn[k], H[k, i], H[k+1, i])

                H[i, i], H[i+1, i], sn[i], cn[i] = generate_givens(
                    H[i, i], H[i+1, i])
                g[i], g[i+1] = apply_givens(sn[i], cn[i], g[i], g[i+1])

                # set L2 norm of residual and output relative residual if necessary
                beta = abs(g[i+1])
                write_history(self.out_file, iters, beta, norm0)

            # end nested FGMRES
            ###################

            # calculate U_new = (Z - U B)R^{-1} g
            # first, solve to get y = R^{-1} g
            y[:i] = numpy.linalg.solve(H[:i, :i], g[:i])
            #y[:i] = g[:i]
            #for k in xrange(i-1,-1,-1):
            #    y[k] /= self.H[k,k]
            #    for k2 in xrange(i-2,-1,-1):
            #        y[k] -= self.H[k2,k]*y[k]
            U_new.equals(0.0)
            for k in xrange(i):
                U_new.equals_ax_p_by(1.0, U_new, y[k], Z[k])
            # update U_new -= U * B
            for k in xrange(self.num_stored):
                tmp = 0.0
                for k2 in xrange(i):
                    tmp += self.B[k,k2]*y[k2]
                U_new.equals_ax_p_by(1.0, U_new, -tmp, self.U[k])

            # finished with g, so undo rotations to find C_new
            y[:i] = g[:i]
            y[i] = 0.0
            for k in xrange(i-1,-1,-1):
                y[k], y[k+1] = apply_givens(-sn[k], cn[k], y[k], y[k+1])
            C_new.equals(0.0)
            for k in xrange(i+1):
                C_new.equals_ax_p_by(1.0, C_new, y[k], W[k])

            # normalize and scale new vectors and update solution and res
            alpha = 1.0/C_new.norm2()
            C_new.divide_by(alpha)
            U_new.divide_by(alpha)
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
                self.C[ptr] = C_new
                self.U[ptr] = U_new

            # get new residual norm; should be the same as the last iter in FGMRES
            beta = res.norm2

            if beta < self.rel_tol*norm0:
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
