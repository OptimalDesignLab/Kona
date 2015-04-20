import sys
import numpy

from kona.options import get_opt

class BaseHessian(object):
    """
    Abstract matrix object that defines the Hessian of an optimization problem.

    Parameters
    ----------
    vector_factory : VectorFactory
    optns : dict, optional
    out_file : file, optional

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
        in_vec : KonaVector
            Vector that gets multiplied with the inverse Hessian.
        out_vec : KonaVector
            Vector that stores the result of the operation.
        rel_tol : float, optional
            Convergence tolerance for the operation.
        """
        raise NotImplementedError # pragma: no cover

class QuasiNewtonApprox(BaseHessian):
    """ Base class for quasi-Newton approximations of the Hessian

    Attributes
    ----------
    max_stored : int
        Maximum number of corrections stored.
    norm_init : float
        Initial norm of design component of gradient.
    init_hessian : KonaVector
        Initial (diagonal) Hessian approximation (stored as a vector).
    s_list : list of KonaVector
        Difference between subsequent solutions: :math:`s_k = x_{k+1} - x_k`
    y_list : list of KonaVector
        Difference between subsequent gradients: :math:`y_k = g_{k+1} - g_k`
    """

    def __init__(self, vector_factory, optns, out_file=sys.stdout):
        super(QuasiNewtonApprox, self).__init__(vector_factory, optns, out_file)
        self.max_stored = get_opt(optns, 10, 'max_stored')

        self.norm_init = 1.0
        self.s_list = []
        self.y_list = []

    def add_correction(self, s_new, y_new):
        """
        Adds a new correction to the Hessian approximation.

        Parameters
        ----------
        s_new : KonaVector
            Difference between subsequent solutions.
        y_new : KonaVector
            Difference between subsequent gradients.
        """
        raise NotImplementedError # pragma: no cover
