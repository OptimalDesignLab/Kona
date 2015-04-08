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
    def __init__(self, memory, transposed=False):
        self._memory = memory
        self._solver = self._memory.solver
        self._transposed = transposed

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

    def _check_type(self, vector, reference):
        if not isinstance(vector, reference):
            raise TypeError('KonaMatrix() >> ' + \
                            'Wrong vector type. Must be a %s.' % reference)

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
        return self.__class__(self._solver, True)

class dRdX(KonaMatrix):

    def product(self, in_vec, out_vec):
        if not self._transposed:
            self._check_type(in_vec, PrimalVector)
            self._check_type(out_vec, StateVector)
            self._solver.multiply_jac_d(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
        else:
            self._check_type(in_vec, StateVector)
            self._check_type(out_vec, PrimalVector)
            self._solver.multiply_tjac_d(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)

class dRdU(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_type(in_vec, StateVector)
        self._check_type(out_vec, StateVector)
        if not self._transposed:
            self._solver.multiply_jac_s(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
        else:
            self._solver.multiply_tjac_s(self._primal._data, self._state._data,
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
        self._check_type(solution, StateVector)
        if not self._transposed:
            self._check_type(rhs_vec, StateVector)
            self._solver.solve_linearsys(self._primal._data, self._state.data,
                                         rhs_vec, rel_tol, solution._data)
        else:
            if rhs_vec is not None:
                self._check_type(rhs_vec, StateVector)
            self._solver.solve_adjoint(self._primal._data, self._state._data,
                                       rhs_vec, rel_tol, solution._data)


class dCdX(KonaMatrix):

    def product(self, in_vec, out_vec):
        if not self._transposed:
            self._check_type(in_vec, PrimalVector)
            self._check_type(out_vec, DualVector)
            self._solver.multiply_ceqjac_d(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
        else:
            self._check_type(in_vec, DualVector)
            self._check_type(out_vec, PrimalVector)
            self._solver.multiply_tceqjac_d(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)

class dCdU(KonaMatrix):

    def product(self, in_vec, out_vec):
        if not self._transposed:
            self._check_type(in_vec, StateVector)
            self._check_type(out_vec, DualVector)
            self._solver.multiply_ceqjac_s(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
        else:
            self._check_type(in_vec, DualVector)
            self._check_type(out_vec, StateVector)
            self._solver.multiply_tceqjac_s(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
