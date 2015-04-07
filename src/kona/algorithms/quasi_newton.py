
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
    s_list : list of KonaVectors
        difference between subsequent solutions: .. math:: s_k = x_{k+1} - x_k
    y_list : list of KonaVectors
        difference between subsequent gradients: .. math:: y_k = g_{k+1} - g_k
    """

    def __init__(self, max_stored, vector_factory, out_file=sys.stdout):
        self.max_stored = max_stored
        self.vector_factory = vector_factory
        self.out_file = out_file

        self.norm_init = 1.0
        self.init_hessian = 1.0
        self.s_list = []
        self.y_list = []

        vector_factory.tally(max_stored)

    def set_inverse_Hessian_to_identity(self):
        """
        Set the initial inverse Hessian to the identity matrix.
        """
        self.init_Hessian = 1.0
        
        ones = vector_factory.generate()
        ones.equals(1.0)
        #INCOMPLETE

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
