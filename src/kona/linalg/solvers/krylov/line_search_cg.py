from kona.linalg.solvers.krylov.basic import KrylovSolver

class LineSearchCG(KrylovSolver):
    """
    Line search Conjugate-Gradient method from page 169 of Numerical Optimization 
    2nd edition by Nocedal and Wright.
    """
    def __init__(self, vector_factory, optns=None):
        super(LineSearchCG, self).__init__(vector_factory, optns)

        # set factory and request vectors needed in solve() method
        self.vec_fac.request_num_vectors(5)
        
        # use initial tolerance as benchmark
        self.init_tol = self.rel_tol

    def solve(self, mat_vec, neg_grad, p, precond=None):
        self._validate_options()

        # grab some vectors from memory stack
        z = self.vec_fac.generate()
        r_old = self.vec_fac.generate()
        r = self.vec_fac.generate()
        d = self.vec_fac.generate()
        Bd = self.vec_fac.generate()

        # define initial residual and other scalars
        p.equals(0.0)
        z.equals(0.0)
        Bd.equals(0.0)
        d.equals(neg_grad)
        r.equals(d)
        r.times(-1.)

        # write header and initial point
        norm0 = r.norm2
        write_header(self.out_file, 'Line-search CG', self.rel_tol, norm0)
        write_history(self.out_file, 0, norm0, norm0)

        # START OF BIG FOR LOOP
        #######################
        for i in xrange(self.max_iter):
            
            mat_vec(d, Bd)
            curv = d.inner(Bd)
            
            # check curvature
            if curv <= -1e-8:
                # return steepest descent if negative
                self.out_file.write('# Negative curvature encountered!\n')
                if i == 0:
                    p.equals(neg_grad)
                else:
                    p.equals(z)
                return (None, False)
            
            alpha = r.inner(r)/curv
            z.equals_ax_p_by(1., z, alpha, d)
            r_old.equals(r)
            r.equals_ax_p_by(1., r, alpha, Bd)
            
            res_norm = r.norm2
            write_history(self.out_file, i+1, res_norm, norm0)
            if res_norm/norm0 <= self.rel_tol:
                p.equals(z)
                return (None, False)
            
            beta = r.inner(r)/r_old.inner(r_old)
            d.equals_ax_p_by(-1., r, beta, d)

        #####################
        # END OF BIG FOR LOOP
            
        # if we got here, solver failed to find an answer
        p.equals(z)
        return (None, True)

# imports here to prevent circular errors
from numpy import sqrt
from kona.linalg.solvers.util import write_header, write_history