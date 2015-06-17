import numpy

class KonaMatrix(object):
    """
    An abstract matrix class connected to Kona memory. This class is used to
    define a variety of jacobian matrices and other composite objects
    containing matrix-related methods used in optimization tasks.

    Parameters
    ----------
    primal : PrimalVector
    state : StateVector
    transposed : boolean, optional

    Attributes
    ----------
    _primal : PrimalVector
        Primal vector point for linearization.
    _state : StateVector
        State vector point for linearization
    _transposed : boolean
        Flag to determine if the matrix is transposed
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
        in_vec : KonaVector
        out_vec : KonaVector

        Returns
        -------
        out_vec : KonaVector
        """
        raise NotImplementedError

    @property
    def T(self):
        """
        Returns the transposed version of the matrix.

        Returns
        -------
        KonaMatrix-like : Transposed version of the matrix.
        """
        return self.__class__(self._primal, self._state, True)

class dRdX(KonaMatrix):
    """
    Partial jacobian of the system residual with respect to primal variables.
    """
    def product(self, in_vec, out_vec):
        self._check_linearization()
        if not self._transposed:
            # self._check_type(in_vec, PrimalVector)
            # self._check_type(out_vec, StateVector)
            self._solver.multiply_dRdX(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)
        else:
            # self._check_type(in_vec, StateVector)
            # self._check_type(out_vec, PrimalVector)
            self._solver.multiply_dRdX_T(self._primal._data, self._state._data,
                                        in_vec._data, out_vec._data)

class dRdU(KonaMatrix):
    """
    Partial jacobian of the system residual with respect to state variables.
    """
    def product(self, in_vec, out_vec):
        self._check_linearization()
        # self._check_type(in_vec, StateVector)
        # self._check_type(out_vec, StateVector)
        if not self._transposed:
            self._solver.multiply_dRdU(
                self._primal._data, self._state._data,
                in_vec._data, out_vec._data)
        else:
            self._solver.multiply_dRdU_T(
                self._primal._data, self._state._data,
                in_vec._data, out_vec._data)

    def solve(self, rhs_vec, solution, rel_tol=1e-8):
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
        # self._check_type(solution, StateVector)
        # self._check_type(rhs_vec, StateVector)
        if not self._transposed:
            cost = self._solver.solve_linear(
                        self._primal._data, self._state._data,
                        rhs_vec._data, rel_tol, solution._data)
        else:
            cost = self._solver.solve_adjoint(
                        self._primal._data, self._state._data,
                        rhs_vec._data, rel_tol, solution._data)

        self._memory.cost += cost

    def precond(self, in_vec, out_vec):
        if not self._transposed:
            cost = self._solver.apply_precond(
                        self._primal._data, self._state._data,
                        in_vec._data, out_vec._data)
        else:
            cost = self._solver.apply_precond_T(
                        self._primal._data, self._state._data,
                        in_vec._data, out_vec._data)

        self._memory.cost += cost

class dCdX(KonaMatrix):
    """
    Partial jacobian of the constraints with respect to primal variables.
    """
    def product(self, in_vec, out_vec):
        self._check_linearization()
        if not self._transposed:
            # self._check_type(in_vec, PrimalVector)
            # self._check_type(out_vec, DualVector)
            self._solver.multiply_dCdX(self._primal._data, self._state._data,
                                           in_vec._data, out_vec._data)
        else:
            # self._check_type(in_vec, DualVector)
            # self._check_type(out_vec, PrimalVector)
            self._solver.multiply_dCdX_T(self._primal._data, self._state._data,
                                            in_vec._data, out_vec._data)

class dCdU(KonaMatrix):
    """
    Partial jacobian of the constraints with respect to state variables.
    """
    def product(self, in_vec, out_vec):
        self._check_linearization()
        if not self._transposed:
            # self._check_type(in_vec, StateVector)
            # self._check_type(out_vec, DualVector)
            self._solver.multiply_dCdU(self._primal._data, self._state._data,
                                           in_vec._data, out_vec._data)
        else:
            # self._check_type(in_vec, DualVector)
            # self._check_type(out_vec, StateVector)
            self._solver.multiply_dCdU_T(self._primal._data, self._state._data,
                                            in_vec._data, out_vec._data)

class IdentityMatrix(KonaMatrix):
    """
    Simple identity matrix abstraction. Like all identity matrices, this one
    does not do anything particularly useful either.
    """
    def __init__(self, *args, **kwargs):
        pass

    def linearize(self, *args, **kwargs):
        pass

    def product(self, in_vec, out_vec):
        out_vec.equals(in_vec)
