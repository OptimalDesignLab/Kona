import numpy
from numpy import sqrt

from kona.options import BadKonaOption, get_opt
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import EPS, write_header, write_history

class FLECS(KrylovSolver):
    """
    FLexible Equality-Constrained Subproblem Krylov iterative methods.

    .. note::

        Insert SIAM reference here.

    Attributes
    ----------
    vec_factory : tuple of VectorFactory
        A pair of vector factories, one for Primal and one for Dual type.
    primal_factory : VectorFactory
        Factory for PrimalVector objects.
    dual_factory : VectorFactory
        Factory for DualVector objects.
    mu : float
        Quadratic subproblem constraint penalty factor.

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
        self.radius = get_opt(optns, 1e100, 'radius')

        # get quadratic subproblem constraint penalty
        self.mu = get_opt(optns, 0.0, 'mu_init')

        # get scalings
        self.grad_scale = get_opt(optns, 1.0, 'grad_scale')
        self.feas_scale = get_opt(optns, 1.0, 'feas_scale')

        # extract vector factories from the factory array
        self.primal_factory = None
        self.dual_factory = None
        for factory in vec_factory:
            if factory._vec_type is PrimalVector:
                self.primal_factory = factory
            elif factory._vec_type is DualVector:
                self.dual_factory = factory

        # put in memory request
        self.vec_fac.request_num_vectors(2*self.max_iter + 2)

    def _generate_composite(self):
        primal = self.primal_factory.generate()
        dual = self.dual_factory.generate()
        return ReducedKKTVector(primal, dual)

    def _validate_options(self):
        super(FLECS, self).__init__()

        if (self.primal_factory is None) or (self.dual_factory is None):
            raise TypeError('wrong vector factory types')

    def _write_header(self, norm0, res0, grad0, feas0):
        self.out_file.write(
            '# FLECS convergence history\n' + \
            '# residual tolerance target = %e\n'%self.rel_tol + \
            '# initial residual norm     = %e\n'%res0 + \
            '# initial gradient norm     = %e\n'%grad0 + \
            '# initial constraint norm   = %e\n'%feas0
            '# iters' + ' '*5 + \
            ' rel. res.' + ' '*5 + \
            'rel. grad.' + ' '*5 + \
            'rel. feas.' + ' '*5 + \
            'aug. feas.' + ' '*5 + \
            '      pred' + ' '*5 + \
            'pred. aug.' + ' '*5 + \
            '        mu' + '\n'
        )

    def _write_history(self, num_iter, res, grad, feas,
                       feas_aug, pred, pred_aug, mu):
        self.out_file.write(
            '# %5i'%num_iter + ' '*5 + \
            '%10e'%res + ' '*5 + \
            '%10e'%grad + ' '*5 + \
            '%10e'%feas + ' '*5 + \
            '%10e'%feas_aug + ' '*5 + \
            '%10e'%pred + ' '*5 + \
            '%10e'%pred_aug + ' '*5 + \
            '%10e'%mu + '\n
        )

    def solve(self, mat_vec, b, x, precond):
        # validate solver options
        self._validate_options()

        # initialize some work data
        V = []
        Z = []
        g = numpy.zeros(self.max_iter + 1)
        y = numpy.zeros(self.max_iter)
        H = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        VtZ = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        ZtZ_prim = numpy.matrix(numpy.zeros((self.max_iter, self.max_iter)))
        VtZ_prim = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        VtZ_dual = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter)))
        VtV_dual = numpy.matrix(numpy.zeros((self.max_iter + 1, self.max_iter + 1)))
        y_aug, y_mult = numpy.ndarray([])
        iters = 0

        # generate residual vector
        res = self._generate_composite()
        res.equals(b)

        # calculate norm of rhs vector
        grad0 = b._primal.norm2
        feas0 = b._dual.norm2
        norm0 = b.norm2

        # calculate initial (negative) residual and compute its norm
        V.append(self._generate_composite())
        V[0].equals(b)
        V[0]._primal.times(self.grad_scale)
        V[0]._dual.times(self.feas_scale)

        # normalize the residual
        beta = V[0].norm2
        V[0].divide_by(beta)
        VtV_dual[0, 0] = V[0]._dual.inner(V[0]._dual)
        gamma = beta*sqrt(max(VtV_dual[0, 0], 0.0))
        omega = sqrt(max(beta**2 - gamma**2, 0.0))

        # initialize RHS of the reduced system
        g[0] = beta

        # output header information
        self._write_header(norm0, grad0, feas0)
        beta_aug = beta
        gamma_aug = gamma
        res_norm = norm0
        pred = 0.0
        pred_aug = 0.0
        self._write_history(iters, res_norm/norm0, omega/(self.grad_scale*grad0),
            gamma/(self.feas_scale*feas0), gamma_aug/(self.feas_scale*feas0),
            pred, pred_aug)

        # loop over all search directions
        #################################
        self.lin_depend = False
        self.neg_curv = False
        self.trust_active = False
        for i in xrange(self.max_iter):
            # advance iteration counter
            iters += 1

            # precondition V[i] and store results in Z[i]
            Z.append(self._generate_composite())
            precond(V[i], Z[i])

            # add to Krylov subspace
            V.append(self._generate.composite())
            Z[i]._primal.times(self.grad_scale)
            Z[i]._dual.times(self.feas_scale)
            mat_vec(Z[i], V[i+1])
            Z[i]._primal.divide_by(self.grad_scale)
            Z[i]._dual.divide_by(self.feas_scale)
            V[i+1]._primal.times(self.grad_scale)
            V[i+1]._dual.times(self.feas_scale)

            # modified Gram-Schmidt orthonogalization
            try:
                raise NotImplementedError
                mod_gram_schmidt(i, H, V)
            except numpy.linalg.LinAlgError:
                self.lin_depend = True

            # compute new row and column of the VtZ matrix
            for k in xrange(i+1):
                VtZ_prim[k, i] = V[k]._primal.inner(Z[i]._primal)
                VtZ_prim[i+1, k] = V[i+1]._primal.inner(Z[k]._primal)

                VtZ_dual[k, i] = V[k]._dual.inner(Z[i]._dual)
                VtZ_dual[i+1, k] = V[i+1]._dual.inner(Z[k]._dual)

                VtZ[k, i] = VtZ_prim[k, i] + VtZ_dual[k, i]
                VtZ[i+1, k] = VtZ_prim[i+1, k] + VtZ_dual[i+1, k]

                ZtZ_prim[k, i] = Z[k]._primal.inner(Z[i]._primal)
                ZtZ_prim[i+1, k] = Z[i+1]._primal.inner(Z[k]._primal)

                VtV_dual[k, i] = V[k]._dual.inner(V[i]._dual)
                VtV_dual[i+1, k] = V[i+1]._dual.inner(V[k]._dual)

            VtV_dual[i+1, i+1] = V[i+1]._dual.inner(V[i+1]._dual)

            # solve the reduced problems and compute the residual
            beta, gamma, omega, beta_aug, gamma_aug, pred, pred_aug = \
                self.solve_subspace_problems(i+1, H, g, ZtZ_prim, VtZ,
                    VtZ_prim, VtZ_dual, VtV_dual, y, y_aug, y_mult)

            # calculate the new residual norm
            res_norm = (gamma/self.feas_scale)**2 + \
                (beta**2 - gamma**2)/(self.grad_scale**2)
            res_norm = sqrt(max(res_norm, 0.0))

            # write convergence history
            self._write_history(iters, res_norm/norm0, omega/(self.grad_scale*grad0),
                gamma/(self.feas_scale*feas0), gamma_aug/(self.feas_scale*feas0),
                pred, pred_aug)

            # check for convergence
            if (gamma < self.rel_tol*self.feas_scale*feas0) and \
            (omega < self.rel_tol*self.grad_scale*grad0):
                break

        #########################################
        # finished looping over search directions
        if self.neg_curv:
            self.out_file.write('# negative curvature suspected\n')
        if self.trust_active:
            self.out_file.write('# trust-radius constraint active\n')

        # compute solution: augmented-Lagrangian step for primal, FGMRES for dual
        x.equals(0.0)
        for k in xrange(iters):
            x._primal.equals_ax_p_by(1.0, x._primal, y_aug[k], Z[k]._primal)
            x._dual.equals_ax_p_by(1.0, x._dual, y_mult[k], Z[k]._dual)

        # scale the solution
        x._primal.times(self.grad_scale)
        x._dual.times(self.feas_scale)

        # check residual
        if self.check_res:
            # calculate true residual for the solution
            V[0].equals(0.0)
            for k in xrange(iters):
                V[0].equals_ax_p_by(1.0, V[0], y_mult[k], Z[k])
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
            computed_feas = gamma/self.feas_scale
            if (abs(true_feas - computed_feas) > 0.01*self.rel_tol*feas0):
                out_data = (true_feas - computed_feas)/feas0
                self.out_file.write(
                    '# WARNING in FLECS: true constraint norm and ' + \
                    'calculated constraint norm do not agree.\n' + \
                    '# (feas_true - feas_comp)/feas0 = %e\n'%out_data
                )
