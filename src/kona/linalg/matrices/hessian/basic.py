
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
    def __init__(self, vector_factory, optns=None):
        # get options dict
        if optns is None:
            self.optns = {}
        else:
            assert type(optns) is dict, "Invalid options! Must be a dictionary."
            self.optns = optns

        # get output file
        self.out_file = get_opt(self.optns, sys.stdout, 'out_file')
        if isinstance(self.out_file, str):
            try:
                _memory = self.vec_fac._memory
            except Exception:
                _memory = self.vec_fac[0]._memory
            self.out_file = _memory.open_file(self.out_file)

        # get references to individual factories
        self.vec_fac = vector_factory
        self.primal_factory = None
        self.state_factory = None
        self.eq_factory = None
        self.ineq_factory = None
        if type(self.vec_fac) is list:
            for factory in self.vec_fac:
                if factory is not None:
                    if factory._vec_type is DesignVector:
                        self.primal_factory = factory
                    elif factory._vec_type is StateVector:
                        self.state_factory = factory
                    elif factory._vec_type is DualVectorEQ:
                        self.eq_factory = factory
                    elif factory._vec_type is DualVectorINEQ:
                        self.ineq_factory = factory
                    else:
                        raise TypeError('Invalid vector factory!')

    def product(self, in_vec, out_vec):
        """
        Applies the Hessian itself to the input vector.

        Parameters
        ----------
        in_vec : KonaVector
            Vector that gets multiplied with the inverse Hessian.
        out_vec : KonaVector
            Vector that stores the result of the operation.
        """
        raise NotImplementedError

    def solve(self, in_vec, out_vec, rel_tol=1e-15):
        """
        Applies the inverse of the approximate Hessian to the input vector.

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

    def __init__(self, vector_factory, optns={}):
        assert isinstance(vector_factory, VectorFactory), \
            "LimitedMemoryBFGS() >> Invalid vector factory!"
        super(QuasiNewtonApprox, self).__init__(vector_factory, optns)
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

# imports at the bottom to prevent circular import errors
import sys
from kona.options import get_opt
from kona.linalg.memory import VectorFactory
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ