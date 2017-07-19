
class KonaMatrix(object):
    """
    An abstract matrix class connected to Kona memory. This class is used to
    define a variety of jacobian matrices and other composite objects
    containing matrix-related methods used in optimization tasks.

    Parameters
    ----------
    primal : DesignVector
    state : StateVector
    transposed : boolean, optional

    Attributes
    ----------
    _design : PrimalVector
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
        primal : DesignVector or CompositePrimalVector
        state : StateVector
        """
        if isinstance(primal, CompositePrimalVector):
            self._design = primal.design
        else:
            self._design = primal
        self._state = state
        if self._design._memory != self._state._memory:
            raise RuntimeError('KonaMatrix() >> ' +
                               'Vectors live on different memory!')
        else:
            self._memory = self._design._memory
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
        return self.__class__(self._design, self._state, True)

class dRdX(KonaMatrix):
    """
    Partial jacobian of the system residual with respect to primal variables.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            if isinstance(in_vec, CompositePrimalVector):
                in_design = in_vec.design
            elif isinstance(in_vec, DesignVector):
                in_design = in_vec
            else:
                raise TypeError(
                    "Invalid multiplying vector: " +
                    "must be DesignVector or CompositePrimalVector!")
            assert isinstance(out_vec, StateVector), \
                "Invalid output vector: must be StateVector!"
            self._solver.multiply_dRdX(
                self._design.base.data, self._state.base,
                in_design.base.data, out_vec.base)
        else:
            assert isinstance(in_vec, StateVector), \
                "Invalid multiplying vector: must be StateVector!"
            if isinstance(out_vec, CompositePrimalVector):
                out_design = out_vec.design
            elif isinstance(out_vec, DesignVector):
                out_design = out_vec
            else:
                raise TypeError(
                    "Invalid output vector: " +
                    "must be DesignVector or CompositePrimalVector!")
            out_design.base.data = self._solver.multiply_dRdX_T(
                self._design.base.data, self._state.base,
                in_vec.base)

class dRdU(KonaMatrix):
    """
    Partial jacobian of the system residual with respect to state variables.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        assert isinstance(in_vec, StateVector), \
            "Invalid multiplying vector: must be StateVector!"
        assert isinstance(out_vec, StateVector), \
            "Invalid output vector: must be StateVector!"
        if not self._transposed:
            self._solver.multiply_dRdU(
                self._design.base.data, self._state.base,
                in_vec.base, out_vec.base)
        else:
            self._solver.multiply_dRdU_T(
                self._design.base.data, self._state.base,
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
        assert isinstance(solution, StateVector), \
            "Invalid solution vector: must be StateVector!"
        assert isinstance(rhs_vec, StateVector), \
            "Invalid RHS vector: must be StateVector!"
        converged = False
        solution.equals(0.0)
        if not self._transposed:
            cost = self._solver.solve_linear(
                self._design.base.data, self._state.base,
                rhs_vec.base, rel_tol, solution.base)
        else:
            cost = self._solver.solve_adjoint(
                self._design.base.data, self._state.base,
                rhs_vec.base, rel_tol, solution.base)
        self._memory.cost += abs(cost)
        if cost >= 0:
            converged = True
        return converged

    def precond(self, in_vec, out_vec):
        assert isinstance(in_vec, StateVector), \
            "Invalid multiplying vector: must be StateVector!"
        assert isinstance(out_vec, StateVector), \
            "Invalid output vector: must be StateVector!"
        if not self._transposed:
            self._solver.apply_precond(
                self._design.base.data, self._state.base,
                in_vec.base, out_vec.base)
        else:
            self._solver.apply_precond_T(
                self._design.base.data, self._state.base,
                in_vec.base, out_vec.base)
        self._memory.cost += 1

class dCEQdX(KonaMatrix):
    """
    Partial jacobian of the equality constraints with respect to design vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            if isinstance(in_vec, CompositePrimalVector):
                in_design = in_vec.design
            elif isinstance(in_vec, DesignVector):
                in_design = in_vec
            else:
                raise AssertionError(
                    "Invalid multiplying vector: " +
                    "must be DesignVector or CompositePrimalVector!")
            assert isinstance(out_vec, DualVectorEQ), \
                "Invalid output vector: must be DualVectorEQ!"
            out_vec.base.data[:] = self._solver.multiply_dCEQdX(
                self._design.base.data, self._state.base,
                in_design.base.data)
        else:
            assert isinstance(in_vec, DualVectorEQ), \
                "Invalid multiplying vector: must be DualVectorEQ!"
            if isinstance(out_vec, CompositePrimalVector):
                out_design = out_vec.design
            elif isinstance(out_vec, DesignVector):
                out_design = out_vec
            else:
                raise AssertionError(
                    "Invalid output vector: " +
                    "must be DesignVector or CompositePrimalVector!")
            out_design.base.data[:] = self._solver.multiply_dCEQdX_T(
                self._design.base.data, self._state.base,
                in_vec.base.data)

class dCEQdU(KonaMatrix):
    """
    Partial jacobian of the equality constraints with respect to state vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            assert isinstance(in_vec, StateVector), \
                "Invalid multiplying vector: must be StateVector!"
            assert isinstance(out_vec, DualVectorEQ), \
                "Invalid output vector: must be DualVectorEQ!"
            out_vec.base.data[:] = self._solver.multiply_dCEQdU(
                self._design.base.data, self._state.base,
                in_vec.base)
        else:
            assert isinstance(in_vec, DualVectorEQ), \
                "Invalid multiplying vector: must be DualVectorEQ!"
            assert isinstance(out_vec, StateVector), \
                "Invalid output vector: must be StateVector!"
            self._solver.multiply_dCEQdU_T(
                self._design.base.data, self._state.base,
                in_vec.base.data, out_vec.base)

class dCINdX(KonaMatrix):
    """
    Partial jacobian of the inequality constraints with respect to design vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            if isinstance(in_vec, CompositePrimalVector):
                in_design = in_vec.design
            elif isinstance(in_vec, DesignVector):
                in_design = in_vec
            else:
                raise AssertionError(
                    "Invalid multiplying vector: " +
                    "must be DesignVector or CompositePrimalVector!")
            assert isinstance(out_vec, DualVectorINEQ), \
                "Invalid output vector: must be DualVectorINEQ!"
            out_vec.base.data[:] = self._solver.multiply_dCINdX(
                self._design.base.data, self._state.base,
                in_design.base.data,)
        else:
            assert isinstance(in_vec, DualVectorINEQ), \
                "Invalid multiplying vector: must be DualVectorINEQ!"
            if isinstance(out_vec, CompositePrimalVector):
                out_design = out_vec.design
            elif isinstance(out_vec, DesignVector):
                out_design = out_vec
            else:
                raise AssertionError(
                    "Invalid output vector: " +
                    "must be DesignVector or CompositePrimalVector!")
            out_design.base.data[:] = self._solver.multiply_dCINdX_T(
                self._design.base.data, self._state.base,
                in_vec.base.data)

class dCINdU(KonaMatrix):
    """
    Partial jacobian of the inequality constraints with respect to state vars.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            assert isinstance(in_vec, StateVector), \
                "Invalid multiplying vector: must be StateVector!"
            assert isinstance(out_vec, DualVectorINEQ), \
                "Invalid output vector: must be DualVectorINEQ!"
            out_vec.base.data[:] = self._solver.multiply_dCINdU(
                self._design.base.data, self._state.base,
                in_vec.base)
        else:
            assert isinstance(in_vec, DualVectorINEQ), \
                "Invalid multiplying vector: must be DualVectorINEQ!"
            assert isinstance(out_vec, StateVector), \
                "Invalid output vector: must be StateVector!"
            self._solver.multiply_dCINdU_T(
                self._design.base.data, self._state.base,
                in_vec.base.data, out_vec.base)

class dCdX(KonaMatrix):
    """
    Combined partial constraint jacobian matrix that can do both equality and
    inequality products depending on what input vectors are provided.
    """
    def product(self, in_vec, out_vec):
        assert self._linearized
        if not self._transposed:
            if isinstance(out_vec, CompositeDualVector):
                dCEQdX(self._design, self._state).product(
                    in_vec, out_vec.eq)
                dCINdX(self._design, self._state).product(
                    in_vec, out_vec.ineq)
            elif isinstance(out_vec, DualVectorEQ):
                dCEQdX(self._design, self._state).product(
                    in_vec, out_vec)
            elif isinstance(out_vec, DualVectorINEQ):
                dCINdX(self._design, self._state).product(
                    in_vec, out_vec)
            else:
                raise AssertionError(
                    "Invalid output vector: " +
                    "must be DualVectorEQ, DualVectorINEQ " +
                    "or CompositeDualVector!")
        else:
            if isinstance(out_vec, CompositePrimalVector):
                out_design = out_vec.design
            elif isinstance(out_vec, DesignVector):
                out_design = out_vec
            else:
                raise AssertionError(
                    "Invalid output vector: " +
                    "must be DesignVector or CompositePrimalVector!")
            if isinstance(in_vec, CompositeDualVector):
                out_design.base.data[:] = self._solver.multiply_dCEQdX_T(
                    self._design.base.data, self._state.base,
                    in_vec.eq.base.data)
                out_design.base.data[:] += self._solver.multiply_dCINdX_T(
                    self._design.base.data, self._state.base,
                    in_vec.ineq.base.data)
            elif isinstance(in_vec, DualVectorEQ):
                dCEQdX(self._design, self._state).T.product(
                    in_vec, out_vec)
            elif isinstance(in_vec, DualVectorINEQ):
                dCINdX(self._design, self._state).T.product(
                    in_vec, out_vec)
            else:
                raise AssertionError(
                    "Invalid multiplying vector: " +
                    "must be DualVectorEQ, DualVectorINEQ " +
                    "or CompositeDualVector!")


class dCdU(KonaMatrix):
    """
    Combined partial constraint jacobian matrix that can do both equality and
    inequality products depending on what input vectors are provided.
    """
    def product(self, in_vec, out_vec, state_work=None):
        assert self._linearized
        if not self._transposed:
            if isinstance(out_vec, CompositeDualVector):
                dCEQdU(self._design, self._state).product(
                    in_vec, out_vec.eq)
                dCINdU(self._design, self._state).product(
                    in_vec, out_vec.ineq)
            elif isinstance(out_vec, DualVectorEQ):
                dCEQdU(self._design, self._state).product(
                    in_vec, out_vec)
            elif isinstance(out_vec, DualVectorINEQ):
                dCINdU(self._design, self._state).product(
                    in_vec, out_vec)
            else:
                raise AssertionError(
                    "Invalid output vector: " +
                    "must be DualVectorEQ, DualVectorINEQ " +
                    "or CompositeDualVector!")
        else:
            assert isinstance(out_vec, StateVector)
            if isinstance(in_vec, CompositeDualVector):
                assert isinstance(state_work, StateVector), \
                    "Invalid work vector: must be StateVector!"
                dCEQdU(self._design, self._state).T.product(
                    in_vec.eq, out_vec)
                dCINdU(self._design, self._state).T.product(
                    in_vec.ineq, state_work)
                out_vec.plus(state_work)
            elif isinstance(in_vec, DualVectorEQ):
                dCEQdU(self._design, self._state).T.product(
                    in_vec, out_vec)
            elif isinstance(in_vec, DualVectorINEQ):
                dCINdU(self._design, self._state).T.product(
                    in_vec, out_vec)
            else:
                raise AssertionError(
                    "Invalid input vector: " +
                    "must be DualVectorEQ, DualVectorINEQ " +
                    "or CompositeDualVector!")

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
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector