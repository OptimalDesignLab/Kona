import numpy

class Hessian(object):

    def __init__(self, vector_factory, optns=None):
        pass

    def product(self, in_vec, out_vec):
        pass

    def solve(self, rhs_vec, rel_tol, out_vec):
        pass

class QuasiNewtonApprox(Hessian):
    pass

class LimitedMemoryBFGS(QuasiNewtonApprox):
    pass

class LimitedMemorySR1(QuasiNewtonApprox):
    pass

class ReducedKKTMatrix(object):
    pass
