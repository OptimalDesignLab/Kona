
import sys

class QuasiNewton(object):
    """ Base class for quasi-Newton methods

    Attributes
    ----------
    max_stored : int
        maximum number of corrections stored
    out_file : file handle
        output file for diagnostics
    norm_init : float
        initial norm of design component of gradient
    init_hessian : KonaVector
        initial (diagonal) Hessian approximation (stored as a vector)
    s : list of KonaVectors
        difference between subsequent solutions: .. math:: s_k = x_{k+1} - x_k
    y : list of KonaVectors
        difference between subsequent gradients: .. math:: y_k = g_{k+1} - g_k
    """

    def __init__(self, max_stored, vector_factory, out_file=sys.stdout):
        self.max_stored = max_stored
        self.num_stored = 0
        self.out_file = out_file
        self.s = vector_factory.create_list(max_stored)
        self.y = vector_factory.create_list(max_stored)
        self.init_Hessian = vector_factor(1)

    def set_inverse_Hessian_to_identity(self):
        """
        set the initial inverse Hessian to the identity matrix
        """
        self.init_Hessian = 1.0
