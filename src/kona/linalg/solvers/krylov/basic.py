from kona.options import get_opt

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
    def __init__(self, vector_factory, optns=None):
        # save the vector factory
        self.vec_fac = vector_factory

        # set up empty optns dict
        if optns is None:
            self.optns = {}
        else:
            assert type(optns) is dict, "Invalid options! Must be a dictionary."
            self.optns = optns

        # get default options
        self.max_iter = get_opt(self.optns, 10, 'subspace_size')
        self.rel_tol = get_opt(self.optns, 1e-6, 'rel_tol')
        self.check_res = get_opt(self.optns, True, 'check_res')

        # set up the info file
        self.out_file = get_opt(self.optns, 'kona_krylov.dat', 'krylov_file')
        if isinstance(self.out_file, str):
            try:
                _memory = self.vec_fac._memory
            except Exception:
                _memory = self.vec_fac[0]._memory
            self.out_file = _memory.open_file(self.out_file)

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
