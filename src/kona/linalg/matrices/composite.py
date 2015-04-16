import sys
import numpy

class Hessian(object):

    def __init__(self, vector_factory, optns=None, out_file=sys.stdout):
        self.vec_fac = vector_factory
        self.out_file = out_file

    def solve(self, rhs_vec, out_vec, rel_tol=1e-15):
        raise NotImplementedError # pragma: no cover

class ReducedKKTMatrix(object):
    pass
