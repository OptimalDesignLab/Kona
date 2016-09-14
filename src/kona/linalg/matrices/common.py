
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

    def linearize(self, primal, state):
        """
        Store the vector points around which a non-linear matrix should be
        linearized.

        Parameters
        ----------
        primal : PrimalVector
        state : StateVector
        """
        self.primal = primal
        self._state = state
        if self.primal._memory != self._state._memory:
            raise RuntimeError('KonaMatrix() >> ' +
                               'Vectors live on different memory!')
        else:
            self._memory = self.primal._memory
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
        return self.__class__(self.primal, self._state, True)

class dRdX(KonaMatrix):
    """
    Partial jacobian of the system residual with respect to primal variables.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            assert isinstance(in_vec, PrimalVector)
            assert isinstance(out_vec, StateVector)
            self._solver.multiply_dRdX(
                self.primal.base.data, self._state.base,
                in_vec.base.data, out_vec.base)
        else:
            assert isinstance(in_vec, StateVector)
            assert isinstance(out_vec, PrimalVector)
            out_vec.base.data = self._solver.multiply_dRdX_T(
                self.primal.base.data, self._state.base,
                in_vec.base)

class dRdU(KonaMatrix):
    """
    Partial jacobian of the system residual with respect to state variables.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        assert isinstance(in_vec, StateVector)
        assert isinstance(out_vec, StateVector)
        if not self._transposed:
            self._solver.multiply_dRdU(
                self.primal.base.data, self._state.base,
                in_vec.base, out_vec.base)
        else:
            self._solver.multiply_dRdU_T(
                self.primal.base.data, self._state.base,
                in_vec.base, out_vec.base)

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
        bool
            Convergence flag.
        """
        assert self._linearized
        assert isinstance(solution, StateVector)
        assert isinstance(rhs_vec, StateVector)
        converged = False
        if not self._transposed:
            cost = self._solver.solve_linear(
                self.primal.base.data, self._state.base,
                rhs_vec.base, rel_tol, solution.base)
        else:
            cost = self._solver.solve_adjoint(
                self.primal.base.data, self._state.base,
                rhs_vec.base, rel_tol, solution.base)
        if cost >= 0:
            converged = True
            self._memory.cost += cost
        else:
            self._memory.cost -= cost
        return converged

    def precond(self, in_vec, out_vec):
        if not self._transposed:
            self._solver.apply_precond(
                self.primal.base.data, self._state.base,
                in_vec.base, out_vec.base)
        else:
            self._solver.apply_precond_T(
                self.primal.base.data, self._state.base,
                in_vec.base, out_vec.base)
        self._memory.cost += 1

class dCdX(KonaMatrix):

    def product(self, in_vec, out_vec):
        raise NotImplementedError

class dCdU(KonaMatrix):

    def product(self, in_vec, out_vec):
        raise NotImplementedError


class dCEQdX(dCdX):
    """
    Partial jacobian of the equality constraints with respect to design vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            assert isinstance(in_vec, PrimalVector)
            assert isinstance(out_vec, DualVectorEQ)
            out_vec.base.data = self._solver.multiply_dCEQdX(
                self.primal.base.data, self._state.base,
                in_vec.base.data,)
        else:
            assert isinstance(in_vec, DualVectorEQ)
            assert isinstance(out_vec, PrimalVector)
            out_vec.base.data = self._solver.multiply_dCEQdX_T(
                self.primal.base.data, self._state.base,
                in_vec.base.data)

class dCEQdU(dCdU):
    """
    Partial jacobian of the equality constraints with respect to state vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            assert isinstance(in_vec, StateVector)
            assert isinstance(out_vec, DualVectorEQ)
            out_vec.base.data = self._solver.multiply_dCEQdU(
                self.primal.base.data, self._state.base,
                in_vec.base)
        else:
            assert isinstance(in_vec, DualVectorEQ)
            assert isinstance(out_vec, StateVector)
            self._solver.multiply_dCEQdU_T(
                self.primal.base.data, self._state.base,
                in_vec.base.data, out_vec.base)

class dCINdX(dCdX):
    """
    Partial jacobian of the inequality constraints with respect to design vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            assert isinstance(in_vec, PrimalVector)
            assert isinstance(out_vec, DualVectorINEQ)
            out_vec.base.data = self._solver.multiply_dCINdX(
                self.primal.base.data, self._state.base,
                in_vec.base.data,)
        else:
            assert isinstance(in_vec, DualVectorINEQ)
            assert isinstance(out_vec, PrimalVector)
            out_vec.base.data = self._solver.multiply_dCINdX_T(
                self.primal.base.data, self._state.base,
                in_vec.base.data)

class dCINdU(dCdU):
    """
    Partial jacobian of the inequality constraints with respect to state vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            assert isinstance(in_vec, StateVector)
            assert isinstance(out_vec, DualVectorINEQ)
            out_vec.base.data = self._solver.multiply_dCINdU(
                self.primal.base.data, self._state.base,
                in_vec.base)
        else:
            assert isinstance(in_vec, DualVectorINEQ)
            assert isinstance(out_vec, StateVector)
            self._solver.multiply_dCINdU_T(
                self.primal.base.data, self._state.base,
                in_vec.base.data, out_vec.base)

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

# package imports at the bottom to prevent import errors
from kona.linalg.vectors.common import PrimalVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ