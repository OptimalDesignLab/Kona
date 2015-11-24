import numpy

from kona.options import get_opt
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import \
    EPS, write_header, write_history, solve_tri, \
    generate_givens, apply_givens, mod_GS_normalize

class FGMRES(KrylovSolver):
    """
    Flexible Generalized Minimum RESidual solver.
    """

    def __init__(self, vector_factory, optns={}, dual_factory=None):
        super(FGMRES, self).__init__(vector_factory, optns)

        # get relative tolerance
        self.rel_tol = get_opt(optns, 0.5, 'rel_tol')

        # put in memory request
        self.vec_fac.request_num_vectors(2*self.max_iter + 1)
        self.dual_fac = dual_factory
        if self.dual_fac is not None:
            self.dual_fac.request_num_vectors(4*self.max_iter + 2)

    def _generate_vector(self):
        if self.dual_fac is None:
            return self.vec_fac.generate()
        else:
            design = self.vec_fac.generate()
            slack = self.dual_fac.generate()
            primal = CompositePrimalVector(design, slack)
            dual = self.dual_fac.generate()
            return ReducedKKTVector(primal, dual)

    def solve(self, mat_vec, b, x, precond):
        # validate solver options
        self._validate_options()

        # initialize some work data
        W = []
        Z = []
        y = numpy.zeros(self.max_iter)
        g = numpy.zeros(self.max_iter + 1)
        sn = numpy.zeros(self.max_iter + 1)
        cn = numpy.zeros(self.max_iter + 1)
        H = numpy.zeros((self.max_iter + 1, self.max_iter))
        iters = 0

        # calculate norm of rhs vector
        norm0 = b.norm2

        # calculate and store the initial residual
        W.append(self._generate_vector())
        mat_vec(x, W[0])
        W[0].times(-1.)
        W[0].plus(b)
        beta = W[0].norm2

        if (beta <= self.rel_tol*norm0) or (beta < EPS):
            # system is already solved
            self.out_file.write('FMGRES system solved by initial guess.\n')
            return iters, beta

        # normalize the residual
        W[0].divide_by(beta)

        # initialize RHS of reduced system
        g[0] = beta

        # output header information
        write_header(self.out_file, 'FGMRES', self.rel_tol, beta)
        write_history(self.out_file, 0, beta, norm0)

        # BEGIN BIG LOOP
        ################

        lin_depend = False
        for i in xrange(self.max_iter):

            # check convergence and linear dependence
            if lin_depend and (beta > self.rel_tol*norm0):
                raise RuntimeError(
                    'FGMRES: Arnoldi process breakdown: ' +
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

            # try modified Gram-Schmidt orthogonalization
            try:
                mod_GS_normalize(i, H, W)
            except numpy.linalg.LinAlgError:
                lin_depend = True

            # apply old Givens rotations to new column of the Hessenberg matrix
            # then generate new Givens rotation matrix and apply it to the last
            # two elements of H[i, :] and g
            for k in xrange(i):
                H[k, i], H[k+1, i] = apply_givens(
                    sn[k], cn[k], H[k, i], H[k+1, i])

            H[i, i], H[i+1, i], sn[i], cn[i] = generate_givens(
                H[i, i], H[i+1, i])
            g[i], g[i+1] = apply_givens(sn[i], cn[i], g[i], g[i+1])

            # set L2 norm of residual and output relative residual if necessary
            beta = abs(g[i+1])
            write_history(self.out_file, i+1, beta, norm0)

        ##############
        # END BIG LOOP

        # solve the least squares system
        y[:i] = solve_tri(H[:i, :i], g[:i], lower=False)
        for k in xrange(i):
            x.equals_ax_p_by(1.0, x, y[k], Z[k])

        if self.check_res:
            # recalculate explicitly and check final residual
            mat_vec(x, W[0])
            W[0].equals_ax_p_by(1.0, b, -1.0, W[0])
            true_res = W[0].norm2
            self.out_file.write(
                '# FGMRES final (true) residual : ' +
                '|res|/|res0| = %e\n'%(true_res/norm0)
            )
            if abs(true_res - beta) > 0.01*self.rel_tol*norm0:
                self.out_file.write(
                    '# WARNING in FGMRES: true residual norm and ' +
                    'calculated residual norm do not agree.\n' +
                    '# (res - beta)/res0 = %e\n'%((true_res - beta)/norm0)
                )
            return iters, true_res
        else:
            return iters, beta
