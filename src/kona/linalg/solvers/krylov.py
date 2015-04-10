
from kona.options import BadKonaOption

class IKrylov(object):
    pass

class STCG(object):
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
    def __init__(self, vector_factory, optns):
        # set factory and request vectors needed in solve() method
        self.vec_factory = vector_factory
        vec_factory.request_num_vectors(4)
        self.max_iter = optns['max_iter']
        self.stcg_tol = optns['stcg_tol']

    def solve(b, x, mat_vec, precond, optns):
        try:
            radius = optns['radius']
            if raidus < 0:
                raise BadKonaOption(optns, ('radius'))
        except KeyError:
            radius = 1.0

        r = self.vec_factory.generate()
        z = self.vec_factory.generate()
        p = self.vec_factory.generate()
        Ap = self.vec_factory.generate()

        # define initial residual and other scalars
        r.equals(b)
        x.equals(0.0)
        precond.product(r, z)
        p.equals(-z)

        norm0 = r.norm2
        r_dot_z = r.inner(z)
        
        if (optns['proj_cg']):
            norm0 = r_dot_z

            for i in xrange(self.max_iter):
                norm_k = r.norm2
                eps = min(0.1, norm_k/norm0)
                if norm_k <= eps:
                    break

                mat_vec(p, Ap)
                alpha =  r_dot_z/p.inner(Ap)   
                x += alpha*p
                r += alpha*Ap

                precond.product(r, z)
                r_dot_z_new = r.inner(z)
        
                beta = r_dot_z_new/r_dot_z
                p = -z + beta*p

                r_dot_z = r_dot_z_new







