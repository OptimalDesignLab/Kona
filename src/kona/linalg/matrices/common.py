import numpy
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector

class KonaMatrix(object):
    """
    An abstract matrix class connected to Kona memory. This class is used to
    define a variety of jacobian matrices and other composite objects
    containing matrix-related methods used in optimization tasks.

    Attributes
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _solver : UserSolver-like
        User solver object that performs the actual data manipulation.
    _transposed : boolean
        Flag to determine if the matrix is transposed
    _primal : PrimalVector
        Primal vector point for linearization.
    _state : StateVector
        State vector point for linearization

    Parameters
    ----------
    memory : KonaMemory
    transposed : boolean (optional)
    """
    def __init__(self, primal=None, state=None, transposed=False):
        self._memory = None
        self._solver = None
        if primal is None or state is None:
            self._linearized = False
        else:
            self.linearize(primal, state)
        self._transposed = transposed

    def _check_type(self, vector, reference):
        if not isinstance(vector, reference):
            raise TypeError('KonaMatrix() >> ' + \
                            'Wrong vector type. Must be a %s.' % reference)

    def _check_linearization(self):
        if not self._linearized:
            raise RuntimeError('KonaMatrix.product() >> ' + \
                               'Matrix must be linearized first!')

    def linearize(self, primal, state):
        """
        Store the vector points around which a non-linear matrix should be
        linearized.

        Parameters
        ----------
        primal : PrimalVector
        state : StateVector
        """
        self._primal = primal
        self._state = state
        if self._primal._memory != self._state._memory:
            raise RuntimeError('KonaMatrix() >> ' + \
                               'Vectors live on different memory!')
        else:
            self._memory = self._primal._memory
            self._solver = self._memory.solver
        self._linearized = True

    def product(self, in_vec, out_vec):
        """
        Performs a matrix-vector product at the internally stored linearization.

        Parameters
        ----------
        in_vec : KonaVector-like
        out_vec : KonaVector-like

        Returns
        -------
        out_vec : KonaVector-like
        """
        pass

    @property
    def T(self):
        """
        Returns the transposed version of the matrix.
        """
        transposed = self.__class__(self._solver, True)
        transposed.linearize(self._primal, self._state)
        return transposed

class dRdX(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_linearization()
        if not self._transposed:
            self._check_type(in_vec, PrimalVector)
            self._check_type(out_vec, StateVector)
            self._solver.multiply_dRdX(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
        else:
            self._check_type(in_vec, StateVector)
            self._check_type(out_vec, PrimalVector)
            self._solver.multiply_dRdX_T(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)

class dRdU(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_linearization()
        self._check_type(in_vec, StateVector)
        self._check_type(out_vec, StateVector)
        if not self._transposed:
            self._solver.multiply_dRdU(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
        else:
            self._solver.multiply_dRdU_T(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)

    def solve(self, rhs_vec, rel_tol, solution):
        """
        Performs a linear solution with the provided right hand side.

        If the transposed matrix object is used, and the right hand side vector
        is ``None``, then this function performs an adjoint solution.

        Parameters
        ----------
        rhs_vec : StateVector or None
            Right hand side vector for solution.
        rel_tol : float
            Solution tolerance.
        solution : StateVector
            Vector where the result should be stored.

        Returns
        -------
        solution : StateVector
        """
        self._check_linearization()
        self._check_type(solution, StateVector)
        if not self._transposed:
            self._check_type(rhs_vec, StateVector)
            self._solver.solve_linear(self._primal._data, self._state.data,
                                      rhs_vec, rel_tol, solution._data)
        else:
            if rhs_vec is not None:
                self._check_type(rhs_vec, StateVector)
            self._solver.solve_adjoint(self._primal._data, self._state._data,
                                       rhs_vec, rel_tol, solution._data)


class dCdX(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_linearization()
        if not self._transposed:
            self._check_type(in_vec, PrimalVector)
            self._check_type(out_vec, DualVector)
            self._solver.multiply_dCdX(self._primal._data, self._state._data,
                                           in_vec._data, out_vec._data)
        else:
            self._check_type(in_vec, DualVector)
            self._check_type(out_vec, PrimalVector)
            self._solver.multiply_dCdX_T(self._primal._data, self._state._data,
                                            in_vec._data, out_vec._data)

class dCdU(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_linearization()
        if not self._transposed:
            self._check_type(in_vec, StateVector)
            self._check_type(out_vec, DualVector)
            self._solver.multiply_dCdU(self._primal._data, self._state._data,
                                           in_vec._data, out_vec._data)
        else:
            self._check_type(in_vec, DualVector)
            self._check_type(out_vec, StateVector)
            self._solver.multiply_dCdU_T(self._primal._data, self._state._data,
                                            in_vec._data, out_vec._data)

class IdentityMatrix(KonaMatrix):

    def __init__(self, *args, **kwargs):
        pass

    def linearize(self, *args, **kwargs):
        pass

    def product(self, in_vec, out_vec):
        out_vec.equals(in_vec)
