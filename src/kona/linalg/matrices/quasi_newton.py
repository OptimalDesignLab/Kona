import sys

from kona.options import get_opt

from kona.linalg.matrices.composite import Hessian

class QuasiNewtonApprox(Hessian):
    """ Base class for quasi-Newton approximations of the Hessian

    Attributes
    ----------
    max_stored : int
        Maximum number of corrections stored.
    norm_init : float
        Initial norm of design component of gradient.
    init_hessian : KonaVector
        Initial (diagonal) Hessian approximation (stored as a vector).
    s_list : list of KonaVectors
        Difference between subsequent solutions: :math:`s_k = x_{k+1} - x_k`
    y_list : list of KonaVectors
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
        s_new : KonaVector-like
            Difference between subsequent solutions.
        y_new : KonaVector-like
            Difference between subsequent gradients.
        """
        raise NotImplementedError # pragma: no cover
