
import sys

from numpy import sqrt

from kona.options import BadKonaOption, get_opt
from kona.linalg.solvers.util import EPS, write_header, write_history

class KrylovSolver(object):
    pass

class STCG(KrylovSolver):
    """
    Steihaug-Toint Conjugate Gradient (STCG) Krylov iterative method

    Attributes
    ----------
    vec_factory : VectorFactory
    max_iter : int

    Parameters
    ----------
    vector_factory : VectorFactory
    optns : dict
    """
    def __init__(self, vector_factory, optns={}, out_file=sys.stdout):
        # set factory and request vectors needed in solve() method
        self.out_file = out_file
        self.vec_fac = vector_factory
        self.vec_fac.request_num_vectors(5)
        self.max_iter = get_opt(optns, 5000, 'max_iter')
        self.rel_tol = get_opt(optns, 1e-8, 'rel_tol')
        self.radius = get_opt(optns, 1.0, 'radius')
        self.proj_cg = get_opt(optns, False, 'proj_cg')
        self.check_res = get_opt(optns, True, 'check_res')

    def _validate_options(self):
        if self.max_iter <= 0:
            raise ValueError('max_iter must be greater than zero')
        if self.rel_tol <= 0:
            raise ValueError('max_iter must be greater than zero')
        if self.radius < 0:
            raise ValueError('radius must be postive')

    def solve(self, mat_vec, b, x, precond):
        self._validate_options()
        # grab some vectors from memory stack
        r = self.vec_fac.generate()
        z = self.vec_fac.generate()
        p = self.vec_fac.generate()
        Ap = self.vec_fac.generate()
        work = self.vec_fac.generate()
        # define initial residual and other scalars
        r.equals(b)
        x.equals(0.0)
        x_norm2 = x.norm2
        alpha = 0.0

        norm0 = r.norm2
        res_norm2 = norm0
        precond(r, z)
        r_dot_z = r.inner(z)
        if self.proj_cg:
            norm0 = r_dot_z
        p.equals(z)
        Ap.equals(p)
        active = False

        write_header(self.out_file, 'STCG', self.rel_tol, norm0)
        write_history(self.out_file, 0, norm0, norm0)

        # START OF BIG FOR LOOP
        #######################
        for i in xrange(self.max_iter):
            # calculate alpha
            mat_vec(p, Ap)
            alpha = p.inner(Ap)
            # check alpha for non-positive curvature
            if alpha <= 1e-8:
                # direction of non-positive curvature detected
                xp = p.inner(x)
                x2 = x_norm2**2
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
                active = True
                break
            # otherwise we have positive curvature, so let's update alpha
            alpha = r_dot_z/alpha
            # perform x = x + alpha*p
            work.equals(p)
            work.times(alpha)
            x.plus(work)
            # check to see if we hit the trust-region boundary
            x_norm2 = x.norm2
            if x_norm2 > self.radius:
                # we exceeded the trust-region radius, so let's undo the step
                x.minus(work)
                # calculate new step within trust region
                xp = p.inner(x)
                x2 = x_norm2**2
                p2 = p.inner(p)
                tau = (-xp + sqrt(xp**2 - p2*(x2 - self.radius**2)))/p2
                # perform x = x + tau*p
                work.equals(p)
                work.times(tau)
                x.plus(work)
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
                active = True
                break
            # if we got here, we're still inside the trust region
            # compute residual here: r = r + alpha*Ap
            work.equals(Ap)
            work.times(alpha)
            r.plus(work)
            res_norm2 = r.norm2
            # preconditioning
            precond(r, z)
            beta = 1./r_dot_z
            r_dot_z = r.inner(z)
            # check convergence
            if self.proj_cg:
                write_history(self.out_file, i+1, r_dot_z, norm0)
                if r_dot_z < norm0*self.rel_tol:
                    break
            else:
                write_history(self.out_file, i+1, res_norm2, norm0)
                if res_norm2 < norm0*self.rel_tol:
                    break
            # if we didn't converge, update beta
            beta *= r_dot_z
            # perform p = z + beta*p
            p.times(beta)
            p.plus(z)
        #####################
        # END OF BIG FOR LOOP

        # if flagged, perform the residual check
        output_string = ''
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
            self.out_file.write('# STCG final (true) residual : ' + \
                '|res|/|res0| = %e\n'%(res/norm0))
            if failed_res:
                self.out_file.write('# WARNING in STCG.solve(): ' + \
                    'true residual norm and calculated residual norm ' + \
                    'do not agree.\n')
                self.out_file.write('# (res - beta)/res0 = %e\n'%(failed_res))

        # check that the solution satisfies the trust-region
        x_norm2 = x.norm2
        if (x_norm2 - self.radius) > 1e-6:
            raise ValueError('STCG.solve() : solution outside of trust-region')
