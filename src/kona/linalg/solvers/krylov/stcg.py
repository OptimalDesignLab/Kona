from kona.linalg.solvers.krylov.basic import KrylovSolver

class STCG(KrylovSolver):
    """
    Steihaug-Toint Conjugate Gradient (STCG) Krylov iterative method

    Attributes
    ----------
    radius : float
        Trust region radius.
    proj_cg : boolean
    """
    def __init__(self, vector_factory, optns={}, dual_factory=None):
        super(STCG, self).__init__(vector_factory, optns)

        self.rel_tol = get_opt(optns, 1e-5, 'rel_tol')
        self.abs_tol = get_opt(optns, 1e-12, 'abs_tol')

        # set a default trust radius
        # NOTE: the trust radius is set by the optimization algorithm
        self.radius = 1.0

        # get other options
        self.proj_cg = get_opt(optns, False, 'proj_cg')

        # set factory and request vectors needed in solve() method
        self.vec_fac.request_num_vectors(7)
        self.dual_fac = dual_factory
        if self.dual_fac is not None:
            self.dual_fac.request_num_vectors(7)

    def _validate_options(self):
        super(STCG, self)._validate_options()
        if self.radius < 0:
            raise ValueError('radius must be postive')

    def _generate_vector(self):
        if self.dual_fac is None:
            return self.vec_fac.generate()
        else:
            design = self.vec_fac.generate()
            slack = self.dual_fac.generate()
            return CompositePrimalVector(design, slack)

    def solve(self, mat_vec, b, x, precond):
        self._validate_options()

        # grab some vectors from memory stack
        r = self._generate_vector()
        z = self._generate_vector()
        p = self._generate_vector()
        Ap = self._generate_vector()
        work = self._generate_vector()

        # define initial residual and other scalars
        r.equals(b)
        x.equals(0.0)
        x_norm2 = x.norm2

        norm0 = r.norm2
        res_norm2 = norm0
        precond(r, z)
        r_dot_z = r.inner(z)
        if self.proj_cg:
            norm0 = r_dot_z

        p.equals(z)
        # Ap.equals(p)
        active = False

        write_header(self.out_file, 'STCG', self.rel_tol, norm0)
        write_history(self.out_file, 0, norm0, norm0)

        # START OF BIG FOR LOOP
        #######################
        for i in xrange(self.max_iter):

            # to be included from c++
            # if (ptin.get<bool>("dynamic",false))
            #     mat_vec.set_product_tol(ptin.get<double>("tol")*norm0/
            #                   (res_norm2*static_cast<double>(maxiter_)))

            # calculate alpha
            mat_vec(p, Ap)
            alpha = p.inner(Ap)
            # check alpha for non-positive curvature
            if alpha <= -1e-8:
                # direction of non-positive curvature detected
                xp = p.inner(x)
                x2 = x.norm2**2
                p2 = p.inner(p)
                # check if ||p||^2 is significant
                if p2 > EPS:
                    # calculate tau
                    tau = (-xp + sqrt(xp**2 - p2*(x2 - self.radius**2)))/p2
                    # perform x = x + tau*p
                    work.equals(p)
                    work.times(tau)
                    x.plus(work)
                    # perform r = r - tau*Ap
                    work.equals(Ap)
                    work.times(-tau)
                    r.plus(work)

                    if self.proj_cg:
                        precond(r, z)
                        r_dot_z = r.inner(z)
                # update residual norm
                res_norm2 = r.norm2
                # write data
                if self.proj_cg:
                    write_history(self.out_file, i+1, r_dot_z, norm0)
                else:
                    write_history(self.out_file, i+1, res_norm2, norm0)
                # mark trust-region boundary as active and finish solution
                self.out_file.write(
                    '# direction of nonpositive curvature detected: ' +
                    'alpha = %f\n'%alpha)
                active = True
                break
            # otherwise we have positive curvature, so let's update alpha
            alpha = r_dot_z/alpha
            # perform x = x + alpha*p
            work.equals(p)
            work.times(alpha)
            x.plus(work)
            x_norm2 = x.norm2

            # check to see if we hit the trust-region boundary
            if x_norm2 > self.radius:
                # we exceeded the trust-region radius, so let's undo the step
                x.minus(work)
                x_norm2 = x.norm2
                # calculate new step within trust region
                xp = p.inner(x)
                x2 = x.inner(x)
                p2 = p.inner(p)
                tau = (-xp + sqrt(xp**2 - p2*(x2 - self.radius**2)))/p2
                # perform x = x + tau*p
                work.equals(p)
                work.times(tau)
                x.plus(work)
                x_norm2 = x.norm2
                # perform r = r - tau*Ap
                work.equals(Ap)
                work.times(-tau)
                r.plus(work)
                res_norm2 = r.norm2
                # write data
                if self.proj_cg:
                    precond(r, z)
                    r_dot_z = r.inner(z)
                    write_history(self.out_file, i+1, r_dot_z, norm0)
                else:
                    write_history(self.out_file, i+1, res_norm2, norm0)
                # mark the trust-region as active and finish solution
                self.out_file.write('# trust-region boundary encountered\n')
                active = True
                break
            # if we got here, we're still inside the trust region
            # compute residual here: r = r + alpha*Ap
            work.equals(Ap)
            work.times(-alpha)
            r.plus(work)
            res_norm2 = r.norm2
            # preconditioning
            precond(r, z)
            beta = 1./r_dot_z
            r_dot_z = r.inner(z)
            # check convergence
            if self.proj_cg:
                write_history(self.out_file, i+1, r_dot_z, norm0)
                if r_dot_z < norm0*self.rel_tol or r_dot_z < self.abs_tol:
                    break
            else:
                write_history(self.out_file, i+1, res_norm2, norm0)
                if res_norm2 < norm0*self.rel_tol or res_norm2 < self.abs_tol:
                    break
            # if we didn't converge, update beta
            beta *= r_dot_z
            # perform p = z + beta*p
            p.times(beta)
            p.plus(z)
        #####################
        # END OF BIG FOR LOOP

        # compute the predicted decrease in objective
        r.plus(b)
        pred = 0.5*x.inner(r)
        r.minus(b)

        # if flagged, perform the residual check
        failed_res = False
        if self.check_res:
            # get the final residual
            mat_vec(x, r)
            # perform r = b - r
            r.times(-1)
            r.plus(b)
            if self.proj_cg:
                precond(r, z)
                res = r.inner(z)
                if abs(res - r_dot_z) > 0.01*self.rel_tol*norm0:
                    failed_res = True
                    failed_out = (res - r_dot_z)/norm0
            else:
                res = r.norm2
                if abs(res - res_norm2) > 0.01*self.rel_tol*norm0:
                    failed_res = True
                    failed_out = (res - res_norm2)/norm0
            # write the residual check message
            self.out_file.write(
                '# STCG final (true) residual : ' +
                '|res|/|res0| = %e\n'%(res/norm0))
            if failed_res:
                self.out_file.write(
                    '# WARNING in STCG.solve(): ' +
                    'true residual norm and calculated residual norm ' +
                    'do not agree.\n')
                self.out_file.write('# (res - beta)/res0 = %e\n'%(failed_out))

        # check that the solution satisfies the trust-region
        x_norm2 = x.norm2
        if (x_norm2 - self.radius) > 1e-6:
            raise ValueError('STCG.solve() : solution outside of trust-region')

        # return some useful stuff
        return pred, active

# imports here to prevent circular errors
from numpy import sqrt
from kona.options import get_opt
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.solvers.util import EPS, write_header, write_history