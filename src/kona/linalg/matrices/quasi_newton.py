from composite import Hessian
import sys

class QuasiNewtonApprox(Hessian):
    """ Base class for quasi-Newton methods

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
        self.max_stored = optns['max_stored']
        self.vec_fac = vector_factory
        self.out_file = out_file

        self.norm_init = 1.0
        self.s_list = []
        self.y_list = []

    def add_correction(self, s_new, y_new):
        """
        Add the step and change in gradient to the list,
        popping the first entry if it is full.
        """
        if len(self.s_list) == self.max_stored:
            del self.s_list[0]
            del self.y_list[0]

        self.s_list.append(s_new)
        self.y_list.append(y_new)
