import gc
import numpy
from numpy import sqrt

from kona.options import get_opt
from kona.linalg.vectors.common import PrimalVector, DualVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import \
    solve_tri, solve_trust_reduced, eigen_decomp, mod_gram_schmidt, EPS

class FLECS(KrylovSolver):
    """
    FLexible Equality-Constrained Subproblem Krylov iterative solver.

    .. note::

        Insert FLECS paper reference here.

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

        # set a default trust radius
        # NOTE: this will be set by the optimization algorithm later
        self.radius = 1.0

        # set a default quadratic subproblem constraint penalty
        # NOTE: this will be set by the optimization algorithm later
        self.mu = 0.1

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
        self.dual_factory.request_num_vectors(2*(2*self.max_iter + 2))

        # initialize vector holder arrays
        self.V = []
        self.Z = []

    def _generate_vector(self):
        design = self.primal_factory.generate()
        slack = self.dual_factory.generate()
        primal = CompositePrimalVector(design, slack)
        dual = self.dual_factory.generate()
        return ReducedKKTVector(primal, dual)

    def _validate_options(self):
        super(FLECS, self)._validate_options()

        if (self.primal_factory is None) or (self.dual_factory is None):
            raise TypeError('wrong vector factory types')

    def _reset(self):
        # clear out all the vectors stored in V
        # the data goes back to the stack and is used again later
        for vector in self.V:
            del vector._primal._design
            del vector._primal._slack
            del vector._primal
            del vector._dual
            del vector
        self.V = []

        # clear out all vectors stored in Z
        # the data goes back to the stack and is used again later
        for vector in self.Z:
            del vector._primal._design
            del vector._primal._slack
            del vector._primal
            del vector._dual
            del vector
        self.Z = []

        # force garbage collection
        gc.collect()

    def _write_header(self, norm0, grad0, feas0):
        self.out_file.write(
            '#-------------------------------------------------\n' +
            '# FLECS convergence history\n' +
            '# residual tolerance target = %e\n'%self.rel_tol +
            '# initial residual norm     = %e\n'%norm0 +
            '# initial gradient norm     = %e\n'%grad0 +
            '# initial constraint norm   = %e\n'%feas0 +
            '# iters' + ' '*5 +
            'rel. res.   ' + ' '*5 +
            'rel. grad.  ' + ' '*5 +
            'rel. feas.  ' + ' '*5 +
            'aug. feas.  ' + ' '*5 +
            'pred        ' + ' '*5 +
            'pred. aug.  ' + ' '*5 +
            'mu          ' + '\n'
        )

    def _write_history(self, res, grad, feas, feas_aug):
        self.out_file.write(
            '# %5i'%self.iters + ' '*5 +
            '%10e'%res + ' '*5 +
            '%10e'%grad + ' '*5 +
            '%10e'%feas + ' '*5 +
            '%10e'%feas_aug + ' '*5 +
            '%10e'%self.pred + ' '*5 +
            '%10e'%self.pred_aug + ' '*5 +
            '%10e'%self.mu + '\n'
        )

    def apply_correction(self, cnstr, step):
        # perform some aliasing to improve readability
        ZtZ_prim_r = self.ZtZ_prim[0:self.iters, 0:self.iters]
        VtV_dual_r = self.VtV_dual[0:self.iters, 0:self.iters+1]
        H_r = self.H[0:self.iters+1, 0:self.iters]

        # construct and solve the subspace problem
        VtVH = VtV_dual_r.dot(H_r)
        A = ZtZ_prim_r + VtVH + VtVH.T
        rhs = numpy.zeros(self.iters)
        for k in xrange(self.iters):
            rhs[k] = self.V[k]._dual.inner(cnstr)
        self.y = numpy.linalg.solve(A, rhs)

        # construct the design update
        # leave the dual solution untouched
        step._primal.equals(0.0)
        for k in xrange(self.iters):
            step._primal.equals_ax_p_by(
                1.0, step._primal, self.y[k], self.Z[k]._primal)

        # trust radius check
        # NOTE: THIS IS TEMPORARY
        if step._primal.norm2 > 0.5*self.radius:
            step._primal.times(
                0.5*self.radius/step._primal.norm2)

        # compute a new predicted reduction associated with this step
        # self.pred_aug = -self.y.dot(0.5*numpy.array(A.dot(self.y)) + rhs)

    def solve_subspace_problems(self):
        # extract some work arrays
        y_r = self.y[0:self.iters]
        g_r = self.g[0:self.iters+1]
        H_r = self.H[0:self.iters+1, 0:self.iters]
        VtZ_r = self.VtZ[0:self.iters+1, 0:self.iters]
        VtZ_prim_r = self.VtZ_prim[0:self.iters+1, 0:self.iters]
        VtZ_dual_r = self.VtZ_dual[0:self.iters+1, 0:self.iters]
        VtV_dual_r = self.VtV_dual[0:self.iters+1, 0:self.iters+1]
        ZtZ_prim_r = self.ZtZ_prim[0:self.iters, 0:self.iters]

        # solve the reduced (primal-dual) problem (i.e.: FGMRES solution)
        y_r, _ , _, _ = numpy.linalg.lstsq(H_r, g_r)
        # make sure the data is persistent
        self.y[0:self.iters] = y_r[:]

        # compute residuals
        res_red = H_r.dot(y_r) - g_r
        self.beta = numpy.linalg.norm(res_red)
        self.gamma = numpy.inner(res_red, VtV_dual_r.dot(res_red))
        self.omega = -self.gamma
        self.gamma = sqrt(max(self.gamma, 0.0))
        self.omega = sqrt(max(numpy.inner(res_red, res_red) + self.omega, 0.0))

        # find the Hessian of the objective and the Hessian of the augmented
        # Lagrangian in the reduced space
        Hess_red = VtZ_r.T.dot(H_r) - VtZ_dual_r.T.dot(H_r) - \
            H_r.T.dot(VtZ_dual_r)
        VtVH = VtV_dual_r.dot(H_r)
        Hess_aug = self.mu * H_r.T.dot(VtVH)
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
            # NOTE: Numpy Cholesky always returns a lower triangular matrix
            # Since UTU is presumed upper triangular, we transpose it here
            UTU = numpy.linalg.cholesky(ZtZ_prim_r).T
            rhs_aug = solve_tri(UTU.T, rhs_tmp, lower=True)

            for j in xrange(self.iters):
                rhs_tmp[:] = Hess_aug[:,j]
                vec_tmp = solve_tri(UTU.T, rhs_tmp, lower=True)
                Hess_aug[:, j] = vec_tmp[:]

            for j in xrange(self.iters):
                rhs_tmp[:] = Hess_aug[j,:]
                vec_tmp = solve_tri(UTU.T, rhs_tmp, lower=True)
                Hess_aug[j,:] = vec_tmp[:]

            vec_tmp, lamb, self.pred_aug = solve_trust_reduced(
                Hess_aug, rhs_aug, radius_aug)
            self.y_aug = solve_tri(UTU, vec_tmp, lower=False)

        except numpy.linalg.LinAlgError:
            # if Cholesky factorization fails, compute a conservative radius
            eig_vals, _ = eigen_decomp(ZtZ_prim_r)
            radius_aug = self.radius/sqrt(eig_vals[self.iters-1])
            self.y_aug, lamb, self.pred_aug = solve_trust_reduced(
                Hess_aug, rhs_aug, radius_aug)

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

        # compute the predicted reductions in the objective
        self.pred = \
            -0.5*numpy.inner(y_r, numpy.inner(Hess_red, y_r)) \
            + self.g[0]*numpy.inner(VtZ_prim_r[0, 0:self.iters], y_r)

        # determine if negative curvature may be present
        self.neg_curv = False
        if (self.pred_aug - self.pred) > 0.05*abs(self.pred):
            self.neg_curv = True

    def solve(self, mat_vec, b, x, precond):
        # validate solver options
        self._validate_options()

        # reset vector memory for a new fresh solution
        self._reset()

        # initialize some work data
        self.g = numpy.zeros(self.max_iter + 1)
        self.y = numpy.zeros(self.max_iter)
        self.H = numpy.zeros((self.max_iter + 1, self.max_iter))
        self.VtZ = numpy.zeros((self.max_iter + 1, self.max_iter))
        self.ZtZ_prim = numpy.zeros((self.max_iter, self.max_iter))
        self.VtZ_prim = numpy.zeros((self.max_iter + 1, self.max_iter))
        self.VtZ_dual = numpy.zeros((self.max_iter + 1, self.max_iter))
        self.VtV_dual = numpy.zeros((self.max_iter + 1, self.max_iter + 1))
        self.y_aug = numpy.array([])
        self.y_mult = numpy.array([])
        self.iters = 0

        # generate residual vector
        res = self._generate_vector()
        res.equals(b)

        # calculate norm of rhs vector
        grad0 = b._primal.norm2
        feas0 = max(b._dual.norm2, EPS)
        norm0 = b.norm2

        # calculate initial (negative) residual and compute its norm
        self.V.append(self._generate_vector())
        self.V[0].equals(b)
        self.V[0]._primal.times(self.grad_scale)
        self.V[0]._dual.times(self.feas_scale)

        # normalize the residual
        self.beta = self.V[0].norm2
        self.V[0].divide_by(self.beta)
        self.VtV_dual[0, 0] = self.V[0]._dual.inner(self.V[0]._dual)
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
        self._write_history(
            res_norm/norm0,
            self.omega/(self.grad_scale*grad0),
            self.gamma/(self.feas_scale*feas0),
            self.gamma_aug/(self.feas_scale*feas0))

        # loop over all search directions
        #################################
        self.lin_depend = False
        self.neg_curv = False
        self.trust_active = False
        for i in xrange(self.max_iter):
            # advance iteration counter
            self.iters += 1

            # precondition self.V[i] and store results in self.Z[i]
            self.Z.append(self._generate_vector())
            precond(self.V[i], self.Z[i])

            # add to Krylov subspace
            self.V.append(self._generate_vector())
            self.Z[i]._primal.times(self.grad_scale)
            self.Z[i]._dual.times(self.feas_scale)
            mat_vec(self.Z[i], self.V[i+1])
            self.Z[i]._primal.divide_by(self.grad_scale)
            self.Z[i]._dual.divide_by(self.feas_scale)
            self.V[i+1]._primal.times(self.grad_scale)
            self.V[i+1]._dual.times(self.feas_scale)

            # modified Gram-Schmidt orthonogalization
            try:
                mod_gram_schmidt(i, self.H, self.V)
            except numpy.linalg.LinAlgError:
                self.lin_depend = True

            # compute new row and column of the VtZ matrix
            for k in xrange(i+1):
                self.VtZ_prim[k, i] = self.V[k]._primal.inner(self.Z[i]._primal)
                self.VtZ_prim[i+1, k] = self.V[i+1]._primal.inner(
                    self.Z[k]._primal)

                self.VtZ_dual[k, i] = self.V[k]._dual.inner(self.Z[i]._dual)
                self.VtZ_dual[i+1, k] = self.V[i+1]._dual.inner(self.Z[k]._dual)

                self.VtZ[k, i] = self.VtZ_prim[k, i] + self.VtZ_dual[k, i]
                self.VtZ[i+1, k] = self.VtZ_prim[i+1, k] + self.VtZ_dual[i+1, k]

                self.ZtZ_prim[k, i] = self.Z[k]._primal.inner(self.Z[i]._primal)
                self.ZtZ_prim[i, k] = self.Z[i]._primal.inner(self.Z[k]._primal)

                self.VtV_dual[k, i+1] = self.V[k]._dual.inner(self.V[i+1]._dual)
                self.VtV_dual[i+1, k] = self.V[i+1]._dual.inner(self.V[k]._dual)

            self.VtV_dual[i+1, i+1] = self.V[i+1]._dual.inner(self.V[i+1]._dual)

            # solve the reduced problems and compute the residual
            self.solve_subspace_problems()

            # calculate the new residual norm
            res_norm = (self.gamma/self.feas_scale)**2 + \
                (self.beta**2 - self.gamma**2)/(self.grad_scale**2)
            res_norm = sqrt(max(res_norm, 0.0))

            # write convergence history
            self._write_history(
                res_norm/norm0,
                self.omega/(self.grad_scale*grad0),
                self.gamma/(self.feas_scale*feas0),
                self.gamma_aug/(self.feas_scale*feas0))

            # check for convergence
            if (self.gamma < self.rel_tol*self.feas_scale*feas0) and \
                    (self.omega < self.rel_tol*self.grad_scale*grad0):
                break

            # check for breakdown
            if self.lin_depend:
                break

        #########################################
        # finished looping over search directions

        if self.neg_curv:
            self.out_file.write('# negative curvature suspected\n')
        if self.trust_active:
            self.out_file.write('# trust-radius constraint active\n')

        # compute solution: augmented-Lagrangian for primal, FGMRES for dual
        x.equals(0.0)
        for k in xrange(self.iters):
            x._primal.equals_ax_p_by(
                1.0, x._primal, self.y_aug[k], self.Z[k]._primal)
            x._dual.equals_ax_p_by(
                1.0, x._dual, self.y_mult[k], self.Z[k]._dual)

        # scale the solution
        x._primal.times(self.grad_scale)
        x._dual.times(self.feas_scale)

        # check residual
        if self.check_res:
            # calculate true residual for the solution
            self.V[0].equals(0.0)
            for k in xrange(self.iters):
                self.V[0].equals_ax_p_by(
                    1.0, self.V[0], self.y_mult[k], self.Z[k])
            mat_vec(self.V[0], res)
            res.equals_ax_p_by(1.0, b, -1.0, res)
            true_res = res.norm2
            true_feas = res._dual.norm2
            # print residual information
            out_data = true_res/norm0
            self.out_file.write(
                '# FLECS final (true) rel. res.  :   ' +
                '|res|/|res0| = %e\n'%out_data
            )
            # print constraint information
            out_data = true_feas/feas0
            self.out_file.write(
                '# FLECS final (true) rel. feas. : ' +
                '|feas|/|feas0| = %e\n'%out_data
            )
            # print warning for residual disagreement
            if (abs(true_res - res_norm) > 0.01*self.rel_tol*norm0):
                out_data = (true_res - res_norm)/norm0
                self.out_file.write(
                    '# WARNING in FLECS: true residual norm and ' +
                    'calculated residual norm do not agree.\n' +
                    '# (res - beta)/res0 = %e\n'%out_data
                )
            # print warning for constraint disagreement
            computed_feas = self.gamma/self.feas_scale
            if (abs(true_feas - computed_feas) > 0.01*self.rel_tol*feas0):
                out_data = (true_feas - computed_feas)/feas0
                self.out_file.write(
                    '# WARNING in FLECS: true constraint norm and ' +
                    'calculated constraint norm do not agree.\n' +
                    '# (feas_true - feas_comp)/feas0 = %e\n'%out_data
                )

    def re_solve(self, b, x):
        # calculate norms for the RHS vector
        grad0 = b._primal.norm2
        feas0 = max(b._dual.norm2, EPS)
        norm0 = sqrt(grad0**2 + feas0**2)

        # reset some prior data and re-solve the subspace problem
        self.y_aug = numpy.array([])
        self.y_mult = numpy.array([])
        self.neg_curv = False
        self.trust_active = False
        self.radius /= self.grad_scale
        self.solve_subspace_problems()

        # compute residual norm
        res_norm = (self.gamma/self.feas_scale)**2 + \
            (self.beta**2 + self.gamma**2)/(self.grad_scale**2)
        res_norm = sqrt(max(res_norm, 0.0))

        # write into solution history
        self.out_file.write(
            '#-------------------------------------------------\n' +
            '# FLECS resolving at new radius\n')
        self._write_history(
            res_norm/norm0,
            self.omega/(self.grad_scale*grad0),
            self.gamma/(self.feas_scale*feas0),
            self.gamma_aug/(self.feas_scale*feas0))
        if self.neg_curv:
            self.out_file.write('# negative curvature suspected\n')
        if self.trust_active:
            self.out_file.write('# trust-radius constraint active\n')

        # always use composite-step approach in re-solve
        x.equals(0.0)
        for k in xrange(self.iters):
            x._primal.equals_ax_p_by(
                1.0, x._primal, self.y_aug[k], self.Z[k]._primal)
            x._dual.equals_ax_p_by(
                1.0, x._dual, self.y_mult[k], self.Z[k]._dual)
        x._primal.times(self.grad_scale)
        x._dual.times(self.feas_scale)
