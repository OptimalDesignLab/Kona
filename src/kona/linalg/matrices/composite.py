import sys
import numpy

class Hessian(object):
    """
    Abstract matrix object that defines the Hessian of an optimization problem.

    Attributes
    ----------
    vec_fac : VectorFactory
        Generator for arbitrary KonaVector types.
    out_file : file
        File stream for data output.
    """
    def __init__(self, vector_factory, optns=None, out_file=sys.stdout):
        self.vec_fac = vector_factory
        self.out_file = out_file

    def solve(self, in_vec, out_vec, rel_tol=1e-15):
        """
        Applies the inverse of the approximate Hessian.

        Parameters
        ----------
        in_vec : KonaVector-like
            Vector that gets multiplied with the inverse Hessian.
        out_vec : KonaVector-like
            Vector that stores the result of the operation.
        rel_tol : float (optional)
            Convergence tolerance for the operation.
        """
        raise NotImplementedError # pragma: no cover

class ReducedKKTMatrix(object):
    pass
