import numpy as np
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector, \
                                       objective_value
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU

def augmented_lagrangian(at_kkt, at_state, at_ceq, mu):
    """
    Calculate and return the scalar value of the augmented Lagrangian penalty
    function.

    Parameters
    ----------
    at_kkt : ReducedKKTVector
    at_state : StateVector
    at_ceq : DualVector
    mu : float

    Returns
    -------
    float
    """
    aug_lag = objective_value(at_kkt._primal, at_state)
    aug_lag += at_kkt._dual.inner(at_ceq)
    aug_lag += 0.5 * at_ceq.inner(at_ceq)
    return aug_lag

class CompositeVector(object):
    """
    Base class for all composite vectors.

    Parameters
    ----------
    memory : KonaMemory
    primal_vec : PrimalVector, optional
    state_vec : StateVector, optional
    dual_vec : DualVector, optional

    Attributes
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _primal : PrimalVector or None
        Primal component of the composite vector.
    _state : StateVector or None
        State component of the composite vector.
    _dual_vec : DualVector or None
        Dual component of the composite vector.
    """
    def __init__(self, primal_vec, dual_vec, state_vec=None):

        if isinstance(primal_vec, PrimalVector):
            self._primal = primal_vec
            self._memory = self._primal._memory
        else:
            raise TypeError('CompositeVector() >> ' + \
                            'Unidentified design vector.')

        if isinstance(state_vec, StateVector) or state_vec is None:
            self._state = state_vec
        else:
            raise TypeError('CompositeVector() >> ' + \
                            'Unidentified state vector.')

        if isinstance(dual_vec, DualVector):
            self._dual = dual_vec
        else:
            raise TypeError('CompositeVector() >> ' + \
                            'Unidentified dual vector.')

    def _check_type(self, vec):
        if not isinstance(vec, type(self)):
            raise TypeError('CompositeVector() >> ' + \
                            'Wrong vector type. Must be %s' % type(self))

    def equals(self, rhs):
        """
        Used as the assignment operator.

        If val is a scalar, all vector elements are set to the scalar value.

        If val is a vector, the two vectors are set equal.

        Parameters
        ----------
        rhs : float or CompositeVector
            Right hand side term for assignment.
        """
        if isinstance(rhs, (float, int, np.float64, np.int64, np.float32, np.int32)):
            if self._primal is not None:
                self._primal.equals(rhs)
            if self._state is not None:
                self._state.equals(rhs)
            if self._dual is not None:
                self._dual.equals(rhs)
        else:
            self._check_type(rhs)
            if self._primal is not None:
                self._primal.equals(rhs._primal)
            if self._state is not None:
                self._state.equals(rhs._state)
            if self._dual is not None:
                self._dual.equals(rhs._dual)

    def plus(self, vector):
        """
        Used as the addition operator.

        Adds the incoming vector to the current vector in place.

        Parameters
        ----------
        vector : CompositeVector
            Vector to be added.
        """
        self._check_type(vector)
        if self._primal is not None:
            self._primal.plus(vector._primal)
        if self._state is not None:
            self._state.plus(vector._state)
        if self._dual is not None:
            self._dual.plus(vector._dual)

    def minus(self, vector):
        """
        Used as the subtraction operator.

        Subtracts the incoming vector from the current vector in place.

        Parameters
        ----------
        vector : CompositeVector
            Vector to be subtracted.
        """
        self._check_type(vector)
        if self._primal is not None:
            self._primal.minus(vector._primal)
        if self._state is not None:
            self._state.minus(vector._state)
        if self._dual is not None:
            self._dual.minus(vector._dual)

    def times(self, value):
        """
        Used as the multiplication operator.

        Multiplies the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        if isinstance(value, (float, int, np.float64, np.int64, np.float32, np.int32)):
            if self._primal is not None:
                self._primal.times(value)
            if self._state is not None:
                self._state.times(value)
            if self._dual is not None:
                self._dual.times(value)
        else:
            raise TypeError('CompositeVector.times() >> ' + \
                            'Wrong argument type. Must be FLOAT.')

    def divide_by(self, value):
        """
        Used as the division operator.

        Divides the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        self.times(1./value)

    def equals_ax_p_by(self, a, x, b, y):
        """
        Performs a full a*X + b*Y operation between two vectors, and stores
        the result in place.

        Parameters
        ----------
        a, b : float
            Coefficients for the operation.
        x, y : CompositeVector
            Vectors for the operation
        """
        self._check_type(x)
        self._check_type(y)
        if self._primal is not None:
            self._primal.equals_ax_p_by(a, x._primal, b, y._primal)
        if self._state is not None:
            self._state.equals_ax_p_by(a, x._state, b, y._state)
        if self._dual is not None:
            self._dual.equals_ax_p_by(a, x._dual, b, y._dual)

    def inner(self, vector):
        """
        Computes an inner product with another vector.

        Returns
        -------
        float : Inner product.
        """
        self._check_type(vector)
        total_prod = 0
        if self._primal is not None:
            total_prod += self._primal.inner(vector._primal)
        if self._state is not None:
            total_prod += self._state.inner(vector._state)
        if self._dual is not None:
            total_prod += self._dual.inner(vector._dual)
        return total_prod

    @property
    def norm2(self):
        """
        Computes the L2 norm of the vector.

        Returns
        -------
        float : L2 norm.
        """
        prod = self.inner(self)
        if prod < 0:
            raise ValueError('CompositeVector.norm2 >> ' + \
                             'Inner product is negative!')
        else:
            return np.sqrt(prod)

class ReducedKKTVector(CompositeVector):
    """
    A composite vector representing a combined design and dual vectors.

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _design : PrimalVector
        Design component of the composite vector.
    _dual : DualVector
        Dual components of the composite vector.
    """

    def equals_init_guess(self):
        """
        Sets the KKT vector to the initial guess, using the initial design.
        """
        self._primal.equals_init_design()
        self._dual.equals(0.0)

    def equals_KKT_conditions(self, x, state, adjoint, primal_work):
        """
        Calculates the total derivative of the Lagrangian
        :math:`\\mathcal{L}(x, u) = f(x, u) + \\lambda c(x, u)` with respect to
        :math:`\\begin{pmatrix}x & \\lambda\\end{pmatrix}^T`. This total
        derivative represents the Karush-Kuhn-Tucker (KKT) convergence
        conditions for the optimization problem defined by
        :math:`\\mathcal{L}(x, u)`.

        The full expression of the KKT conditions are:

        .. math::
            \\begin{pmatrix} g(x, u, \lambda, \psi) \\ c(x, u) \\end{pmatrix} = \\begin{pmatrix}\\end{pmatrix}

        Parameters
        ----------
        x : ReducedKKTVector
            Evaluate KKT conditions at this primal-dual point.
        state : StateVector
            Evaluate KKT conditions at this state point.
        adjoint : StateVector
            Evaluate KKT conditions using this adjoint vector.
        primal_work : PrimalVector
            Work vector for intermediate calculations.
        """
        # evaluate lagrangian total derivative
        self._primal.equals_lagrangian_total_gradient(
            x._primal, state, x._dual, adjoint, primal_work)
        # evaluate constraints at the design/state point
        self._dual.equals_constraints(x._primal, state)
