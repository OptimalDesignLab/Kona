import numpy
from numpy import sqrt

from kona.options import BadKonaOption, get_opt
from kona.linalg.vectors.common import PrimalVector, DualVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import EPS, write_header, write_history, \
                                     solve_tri, solve_trust_reduced, \
                                     eigen_decomp, mod_gram_schmidt

class FLECS(KrylovSolver):
    """
    FLexible Equality-Constrained Subproblem Krylov iterative methods.

    .. note::

        Insert SIAM reference here.

    Attributes
    ----------
    primal_factory : VectorFactory
        Factory for PrimalVector objects.
    dual_factory : VectorFactory
        Factory for DualVector objects.
    mu : float
        Quadratic subproblem constraint penalty factor.
    grad_scale : float
        Scaling coefficient for the primal space.
    feas_scale : float
        Scaling coefficient for the dual space.
    lin_depend : boolean
        Flag for Hessian linear dependence.
    neg_curv : boolean
        Flag for negative curvature in the search direction.
    trust_active : boolean
        Flag for trust-region detection.

    Parameters
    ----------
    vector_factories : tuple of VectorFactory
        A pair of vector factories, one for Primal and one for Dual type.
    optns : dict, optional
        Optiona dictionary
    """

    def __init__(self, vector_factories, optns={}):
        super(FLECS, self).__init__(vector_factories, optns)

        self.rel_tol = get_opt(optns, 0.5, 'rel_tol')

        # get trust radius
        self.radius = get_opt(optns, 1.0, 'radius')

        # get quadratic subproblem constraint penalty
        self.mu = get_opt(optns, 1.0, 'mu_init')

        # get scalings
        self.grad_scale = get_opt(optns, 1.0, 'grad_scale')
        self.feas_scale = get_opt(optns, 1.0, 'feas_scale')

        # extract vector factories from the factory array
        self.primal_factory = None
        self.dual_factory = None
        for factory in vector_factories:
            if factory._vec_type is PrimalVector:
                self.primal_factory = factory
            elif factory._vec_type is DualVector:
                self.dual_factory = factory

        # put in memory request
        self.primal_factory.request_num_vectors(2*self.max_iter + 2)
        self.dual_factory.request_num_vectors(2*self.max_iter + 2)

    def _generate_vector(self):
        primal = self.primal_factory.generate()
        dual = self.dual_factory.generate()
        return ReducedKKTVector(primal, dual)

    def _validate_options(self):
        super(FLECS, self)._validate_options()

        if (self.primal_factory is None) or (self.dual_factory is None):
            raise TypeError('wrong vector factory types')

    def _write_header(self, norm0, grad0, feas0):
        self.out_file.write(
            '# FLECS convergence history\n' + \
            '# residual tolerance target = %e\n'%self.rel_tol + \
            '# initial residual norm     = %e\n'%norm0 + \
            '# initial gradient norm     = %e\n'%grad0 + \
            '# initial constraint norm   = %e\n'%feas0 + \
            '# iters' + ' '*5 + \
            ' rel. res.' + ' '*5 + \
            'rel. grad.' + ' '*5 + \
            'rel. feas.' + ' '*5 + \
            'aug. feas.' + ' '*5 + \
            '      pred' + ' '*5 + \
            'pred. aug.' + ' '*5 + \
            '        mu' + '\n'
        )

    def _write_history(self, res, grad, feas, feas_aug):
        self.out_file.write(
            '# %5i'%self.iters + ' '*5 + \
            '%10e'%res + ' '*5 + \
            '%10e'%grad + ' '*5 + \
            '%10e'%feas + ' '*5 + \
            '%10e'%feas_aug + ' '*5 + \
            '%10e'%self.pred + ' '*5 + \
            '%10e'%self.pred_aug + ' '*5 + \
            '%10e'%self.mu + '\n'
        )

    def solve_subspace_problems(self):


        # perform some vector/matrix deep copies to preserve previous data
        y_r = numpy.copy( self.y[ 0:self.iters ] )
        g_r = numpy.copy( self.g[ 0:self.iters+1 ] )
        H_r = numpy.copy( self.H[ 0:self.iters+1, 0:self.iters ])
        VtZ_r = numpy.copy( self.VtZ[ 0:self.iters+1, 0:self.iters ])
        VtZ_prim_r = numpy.copy( self.VtZ_prim[ 0:self.iters+1, 0:self.iters ] )
        VtZ_dual_r = numpy.copy( self.VtZ_dual[ 0:self.iters+1, 0:self.iters ] )
        VtV_dual_r = numpy.copy( self.VtV_dual[ 0:self.iters+1, 0:self.iters+1 ] )
        ZtZ_prim_r = numpy.copy( self.ZtZ_prim[ 0:self.iters, 0:self.iters ] )

        # solve the reduced (primal-dual) problem (i.e.: FGMRES solution)
        y_r, _, _, _ = numpy.linalg.lstsq(H_r, g_r)

        # compute residuals
        res_red = H_r.dot(y_r) - g_r
        self.beta = numpy.linalg.norm(res_red)
        self.gamma = numpy.inner(res_red, VtV_dual_r.dot(res_red))
        self.omega = -self.gamma
        self.gamma = sqrt(max(self.gamma, 0.0))
        self.omega = sqrt(max(numpy.inner(res_red, res_red) + self.omega, 0.0))

        # find the Hessian of the objective and the Hessian of the augmented
        # Lagrangian in the reduced space
        Hess_red = VtZ_r.T.dot(H_r) - VtZ_dual_r.T.dot(H_r) - H_r.T.dot(VtZ_dual_r)
        VtVH = VtV_dual_r.dot(H_r)
        Hess_aug = self.mu*H_r.T.dot(VtVH)
        Hess_aug += Hess_red

        # compute the RHS for the augmented Lagrangian problem
        rhs_aug = numpy.zeros(self.iters)
        for k in xrange(self.iters):
            rhs_aug[k] = -self.g[0]*(self.VtZ_prim[0, k] + self.mu*VtVH[0, k])

        lamb = 0.0
        radius_aug = self.radius
        try:
            # compute the transformation to apply trust-radius directly
            rhs_tmp = numpy.copy(rhs_aug)
            UTU = numpy.linalg.cholesky(ZtZ_prim_r).T
            # NOTE: Numpy Cholesky always returns a lower triangular matrix
            # Since UTU is presumed upper triangular, we transpose it here
            rhs_aug = solve_tri(UTU.T, rhs_tmp, lower=True)

            for j in xrange(self.iters):
                rhs_tmp[:] = Hess_aug[:, j]
                vec_tmp = solve_tri(UTU.T, rhs_tmp, lower=True)
                Hess_aug[:, j] = vec_tmp[:]

            for j in xrange(self.iters):
                rhs_tmp[:] = Hess_aug[j, :]
                vec_tmp = solve_tri(UTU.T, rhs_tmp, lower=True)
                Hess_aug[j, :] = vec_tmp[:]

            vec_tmp, lamb, _ = solve_trust_reduced(Hess_aug, rhs_aug, radius_aug)
            self.y_aug = solve_tri(UTU, vec_tmp)

        except numpy.linalg.LinAlgError:
            # if Cholesky factorization fails, compute a conservative radius
            eig_vals, _ = eigen_decomp(ZtZ_prim_r)
            radius_aug = self.radius/sqrt(eig_vals[self.iters-1])
            self.y_aug, lamb, _ = solve_trust_reduced(Hess_aug, rhs_aug, radius_aug)

        # check if the trust-radius constraint is active
        self.trust_active = False
        if lamb > 0.0:
            self.trust_active = True

        # compute residual norms for the augmented Lagrangian solution
        res_red = H_r.dot(self.y_aug) - g_r
        self.beta_aug = numpy.linalg.norm(res_red)
        self.gamma_aug = numpy.inner(res_red, VtV_dual_r.dot(res_red))
        self.gamma_aug = sqrt(max(self.gamma_aug, 0.0))

        # set the dual reduced-space solution
        self.y_mult.resize(self.iters)
        self.y_mult[:] = y_r[:]

        # compute the predicted reductions in the objective (not penalty func.)
        self.pred = -0.5*numpy.inner(y_r, Hess_red.dot(y_r))
        for k in xrange(self.iters):
            self.pred += self.g[0]*VtZ_prim_r[0, k]*y_r[k]
        self.pred_aug = -0.5*numpy.inner(self.y_aug, Hess_red.dot(self.y_aug))
        for k in xrange(self.iters):
            self.pred_aug += self.g[0]*VtZ_prim_r[0, k]*self.y_aug[k]

        # determine if negative curvature may be present
        self.neg_curv = False
        if (self.pred_aug - self.pred) > 0.05*abs(self.pred):
            self.neg_curv = True

    def solve(self, mat_vec, b, x, precond):
        # validate solver options
        self._validate_options()

        # initialize some work data
        V = []
        Z = []
        self.g = numpy.zeros(self.max_iter + 1)
        self.y = numpy.zeros(self.max_iter)
        self.H = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        self.VtZ = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        self.ZtZ_prim = numpy.matrix(numpy.zeros((self.max_iter, self.max_iter)))
        self.VtZ_prim = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        self.VtZ_dual = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        self.VtV_dual = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter + 1)))
        self.y_aug = numpy.ndarray([])
        self.y_mult = numpy.ndarray([])
        self.iters = 0

        # generate residual vector
        res = self._generate_vector()
        res.equals(b)

        # calculate norm of rhs vector
        grad0 = b._primal.norm2
        feas0 = b._dual.norm2
        norm0 = b.norm2

        # calculate initial (negative) residual and compute its norm
        V.append(self._generate_vector())
        V[0].equals(b)
        V[0]._primal.times(self.grad_scale)
        V[0]._dual.times(self.feas_scale)

        # normalize the residual
        self.beta = V[0].norm2
        V[0].divide_by(self.beta)
        self.VtV_dual[0, 0] = V[0]._dual.inner(V[0]._dual)
        self.gamma = self.beta*sqrt(max(self.VtV_dual[0, 0], 0.0))
        self.omega = sqrt(max(self.beta**2 - self.gamma**2, 0.0))

        # initialize RHS of the reduced system
        self.g[0] = self.beta

        # output header information
        self._write_header(norm0, grad0, feas0)
        self.beta_aug = self.beta
        self.gamma_aug = self.gamma
        res_norm = norm0
        self.pred = 0.0
        self.pred_aug = 0.0
        self._write_history(res_norm/norm0, self.omega/(self.grad_scale*grad0),
            self.gamma/(self.feas_scale*feas0), self.gamma_aug/(self.feas_scale*feas0))

        # loop over all search directions
        #################################
        self.lin_depend = False
        self.neg_curv = False
        self.trust_active = False
        for i in xrange(self.max_iter):
            # advance iteration counter
            self.iters += 1

            # precondition V[i] and store results in Z[i]
            Z.append(self._generate_vector())
            precond(V[i], Z[i])

            # add to Krylov subspace
            V.append(self._generate_vector())
            Z[i]._primal.times(self.grad_scale)
            Z[i]._dual.times(self.feas_scale)
            mat_vec(Z[i], V[i+1])
            Z[i]._primal.divide_by(self.grad_scale)
            Z[i]._dual.divide_by(self.feas_scale)
            V[i+1]._primal.times(self.grad_scale)
            V[i+1]._dual.times(self.feas_scale)

            # modified Gram-Schmidt orthonogalization
            try:
                mod_gram_schmidt(i, self.H, V)
            except numpy.linalg.LinAlgError:
                self.lin_depend = True

            # compute new row and column of the VtZ matrix
            for k in xrange(i+1):
                self.VtZ_prim[k, i] = V[k]._primal.inner(Z[i]._primal)
                self.VtZ_prim[i+1, k] = V[i+1]._primal.inner(Z[k]._primal)

                self.VtZ_dual[k, i] = V[k]._dual.inner(Z[i]._dual)
                self.VtZ_dual[i+1, k] = V[i+1]._dual.inner(Z[k]._dual)

                self.VtZ[k, i] = self.VtZ_prim[k, i] + self.VtZ_dual[k, i]
                self.VtZ[i+1, k] = self.VtZ_prim[i+1, k] + self.VtZ_dual[i+1, k]

                self.ZtZ_prim[k, i] = Z[k]._primal.inner(Z[i]._primal)
                self.ZtZ_prim[i, k] = Z[i]._primal.inner(Z[k]._primal)

                self.VtV_dual[k, i] = V[k]._dual.inner(V[i]._dual)
                self.VtV_dual[i+1, k] = V[i+1]._dual.inner(V[k]._dual)

            self.VtV_dual[i+1, i+1] = V[i+1]._dual.inner(V[i+1]._dual)

            # solve the reduced problems and compute the residual
            self.solve_subspace_problems()

            # calculate the new residual norm
            res_norm = (self.gamma/self.feas_scale)**2 + \
                (self.beta**2 - self.gamma**2)/(self.grad_scale**2)
            res_norm = sqrt(max(res_norm, 0.0))

            # write convergence history
            self._write_history(res_norm/norm0, self.omega/(self.grad_scale*grad0),
                self.gamma/(self.feas_scale*feas0), self.gamma_aug/(self.feas_scale*feas0))

            # check for convergence
            if (self.gamma < self.rel_tol*self.feas_scale*feas0) and \
            (self.omega < self.rel_tol*self.grad_scale*grad0):
                break

        #########################################
        # finished looping over search directions
        if self.neg_curv:
            self.out_file.write('# negative curvature suspected\n')
        if self.trust_active:
            self.out_file.write('# trust-radius constraint active\n')

        # compute solution: augmented-Lagrangian step for primal, FGMRES for dual
        x.equals(0.0)
        for k in xrange(self.iters):
            x._primal.equals_ax_p_by(1.0, x._primal, self.y_aug[k], Z[k]._primal)
            x._dual.equals_ax_p_by(1.0, x._dual, self.y_mult[k], Z[k]._dual)

        # scale the solution
        x._primal.times(self.grad_scale)
        x._dual.times(self.feas_scale)

        # check residual
        if self.check_res:
            # calculate true residual for the solution
            V[0].equals(0.0)
            for k in xrange(self.iters):
                V[0].equals_ax_p_by(1.0, V[0], self.y_mult[k], Z[k])
            mat_vec(V[0], res)
            res.equals_ax_p_by(1.0, b, -1.0, res)
            true_res = res.norm2
            true_feas = res._dual.norm2
            # print residual information
            out_data = true_res/norm0
            self.out_file.write(
                '# FLECS final (true) rel. res.  :   ' + \
                '|res|/|res0| = %e\n'%out_data
            )
            # print constraint information
            out_data = true_feas/feas0
            self.out_file.write(
                '# FLECS final (true) rel. feas. : ' + \
                '|feas|/|feas0| = %e\n'%out_data
            )
            # print warning for residual disagreement
            if (abs(true_res - res_norm) > 0.01*self.rel_tol*norm0):
                out_data = (true_res - res_norm)/norm0
                self.out_file.write(
                    '# WARNING in FLECS: true residual norm and ' + \
                    'calculated residual norm do not agree.\n' + \
                    '# (res - beta)/res0 = %e\n'%out_data
                )
            # print warning for constraint disagreement
            computed_feas = self.gamma/self.feas_scale
            if (abs(true_feas - computed_feas) > 0.01*self.rel_tol*feas0):
                out_data = (true_feas - computed_feas)/feas0
                self.out_file.write(
                    '# WARNING in FLECS: true constraint norm and ' + \
                    'calculated constraint norm do not agree.\n' + \
                    '# (feas_true - feas_comp)/feas0 = %e\n'%out_data
                )
