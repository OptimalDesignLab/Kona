
class IKrylov(object):



class STCG(object)
    """
    Steihaug-Toint Conjugate Gradient (STCG) Krylov iterative method


    """

    def __init__(self, vector_factory, optns):
        # set factory and request vectors needed in solve() method
        self.vec_factory = vector_factory
        vec_factory.request_num_vectors(4)
        max_iter = optns['max_iter']

    def solve(b, x, mat_vec, precond, optns)
        try:
            radius = optns['radius']
            if raidus < 0:
                raise BadKonaOption(optns, ('radius'))
        except KeyError:
            radius = 1.0

        r = self.vec_factory.generate()
        z = self.vec_factory.generate()
        

