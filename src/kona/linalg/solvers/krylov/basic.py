
import sys

from kona.options import BadKonaOption, get_opt

class KrylovSolver(object):
    """
    Base class for all Krylov solvers.

    Parameters
    ----------
    vector_factory : VectorFactory
    optns : dict, optional

    Attributes
    ----------
    vec_factory : VectorFactory
        Used to generate abstracted KonaVector objects.
    max_iter : int
        Maximum iterations for the CG solve.
    rel_tol : float
        Relative residual tolerance for the solution.
    check_res : boolean
        Flag for checking the residual after solution is found
    out_file : file
        File stream for writing convergence data.
    """
    def __init__(self, vector_factory, optns={}):
        self.vec_fac = vector_factory
        self.max_iter = get_opt(optns, 10, 'max_iter')
        self.rel_tol = get_opt(optns, 1e-8, 'rel_tol')
        self.check_res = get_opt(optns, True, 'check_res')
        # set up the info file
        self.out_file = get_opt(optns, 'kona_krylov.dat', 'out_file')
        if isinstance(self.out_file, str):
            self.out_file = open(self.out_file,'w')

    def _validate_options(self):
        if self.max_iter < 1:
            raise ValueError('max_iter must be greater than one')
        if self.rel_tol <= 0:
            raise ValueError('max_iter must be greater than zero')

    def solve(self, mat_vec, b, x, precond):
        """
        Solves the Ax=b linear system iteratively.

        Parameters:
        -----------
        mat_vec : function
            Matrix-vector product for left-hand side matrix A.
        b : KonaVector
            Right-hand side vector.
        x : KonaVector
            Solution vector
        precond : function
            Matrix-vector product for approximate inv(A).
        """
        raise NotImplementedError
