import sys

from composite import Hessian

from kona.options import get_opt

class QuasiNewtonApprox(Hessian):
    """ Base class for quasi-Newton approximations of the Hessian

    Attributes
    ----------
    max_stored : int
        maximum number of corrections stored
    vec_fac: VectoryFactory
        used to declare the number of requested vectors and generate vectors
    out_file : file handle
        output file for diagnostics
    norm_init : float
        initial norm of design component of gradient
    init_hessian : KonaVector
        initial (diagonal) Hessian approximation (stored as a vector)
    s_list : list of KonaVectors
        difference between subsequent solutions: .. math:: s_k = x_{k+1} - x_k
    y_list : list of KonaVectors
        difference between subsequent gradients: .. math:: y_k = g_{k+1} - g_k
    """

    def __init__(self, vector_factory, optns, out_file=sys.stdout):
        super(QuasiNewtonApprox, self).__init__(vector_factory, optns, out_file)
        self.max_stored = get_opt(optns, 10, 'max_stored')

        self.norm_init = 1.0
        self.s_list = []
        self.y_list = []

    def add_correction(self, s_new, y_new): # pragma: no cover
        raise NotImplementedError # pragma: no cover
