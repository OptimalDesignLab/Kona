import numpy as np

from kona.linalg.matrices.common import dRdX, dRdU, dCdX


def objective_value(at_design, at_state):
        """
        Evaluate the objective value the given Primal and State point.

        Parameters
        ----------
        at_design : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.

        Returns
        -------
        float : Objective function value.
        """
        solver = at_design._memory.solver

        if solver != at_state._memory.solver:
            raise MemoryError('objective_value() >> Primal and State ' + \
                              'vectors are not on the same memory manager!')

        return solver.eval_obj(at_design._data, at_state._data)


class KonaVector(object):
    """
    An abstract vector class connected to the Kona memory, containing a
    common set of algebraic member functions. Allows Kona to operate on
    data spaces allocated by the user.

    Attributes
    ----------
    _memory : UserMemory (singleton)
        Pointer to the Kona user memory.
    _data : BaseVector or derivative
        User defined vector object that contains data and operations on data.

    Parameters
    ----------
    user_vector : BaseVector or derivative
        User defined vector object that contains data and operations on data.
    memory_obj : UserMemory (singleton)
        Pointer to the Kona user memory.
    """

    def __init__(self, memory_obj, user_vector=None):
        self._memory = memory_obj
        self._data = user_vector

    def __del__(self):
        self._memory.push_vector(type(self), self._data)


    def _check_type(self, vector):
        if not isinstance(vector, type(self)):
            raise TypeError('Vector type mismatch. Must be %s' % type(self))

    def equals(self, val):
        """
        Used as the assignment operator.

        If val is a scalar, all vector elements are set to the scalar value.

        If val is a vector, the two vectors are set equal.

        Parameters
        ----------
        val : float or KonaVector derivative
            Right hand side term for assignment.
        """
        if isinstance(val, (float, np.float32, np.float64, int, np.int32, np.int64)):
            self._data.equals_value(val)
        else:
            self._check_type(val)
            self._data.equals_vector(val._data)

    def plus(self, vector):
        """
        Used as the addition operator.

        Adds the incoming vector to the current vector in place.

        Parameters
        ----------
        vector : KonaVector derivative
            Vector to be added.
        """
        self._check_type(vector)
        self._data.plus(vector._data)

    def minus(self, vector):
        """
        Used as the subtraction operator.

        Subtracts the incoming vector from the current vector in place.

        Parameters
        ----------
        vector : KonaVector derivative
            Vector to be subtracted.
        """
        if vector == self: # special case...
            self.equals(0)

        self._check_type(vector)
        self._data.times(-1.)
        self._data.plus(vector._data)
        self._data.times(-1.)

    def times(self, value):
        """
        Used as the multiplication operator.

        Multiplies the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        if isinstance(value, (float, np.float32, np.float64, int, np.int32, np.int64)):
            self._data.times(value)
        else:
            raise TypeError('Argument must be a float.')

    def divide_by(self, val):
        """
        Used as the multiplication operator.

        Multiplies the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        self.times(1./val)

    def equals_ax_p_by(self, a, X, b, Y):
        """
        Performs a full a*X + b*Y operation between two vectors, and stores
        the result in place.

        Parameters
        ----------
        a, b : float
            Coefficients for the operation.
        x, y : KonaVector or derivative
            Vectors for the operation
        """
        self._check_type(X)
        self._check_type(Y)
        self._data.equals_ax_p_by(a, X._data, b, Y._data)

    def inner(self, vector):
        """
        Computes an inner product with another vector.

        Returns
        -------
        float : Inner product.
        """
        self._check_type(vector)
        return self._data.inner(vector._data)

    @property
    def norm2(self): # this takes the L2 norm of the vector
        """
        Computes the L2 norm of the vector.

        Returns
        -------
        float : L2 norm.
        """
        prod = self.inner(self)
        return np.sqrt(prod)

class PrimalVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific
    to design vectors.
    """

    def restrict_target_state(self):
        """
        Set target state variables to zero.

        Used only for IDF problems.
        """
        self._memory.solver.restrict_design(0, self._data)

    def restrict_real_design(self):
        """
        Set design variables to zero.

        Used only for IDF problems.
        """
        self._memory.solver.restrict_design(1, self._data)

    def convert(self, dual_vector):
        """
        Copy target state variables from the dual space into the design space.

        Used only for IDF problems.

        Parameters
        ----------
        dual_vector : DualVector
            Source vector for target state variable data.
        """
        self._memory.solver.copy_dual_to_targstate(dual_vector._data,
                                                   self._data)

    def equals_init_design(self):
        """
        Sets this vector equal to the initial design point.
        """
        self._memory.solver.init_design(self._data)

    def equals_objective_partial(self, at_primal, at_state):
        """
        Computes in-place the partial derivative of the objective function with
        respect to design variables.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        self._memory.solver.eval_dFdX(at_primal._data,
                                      at_state._data,
                                      self._data)

    def equals_total_gradient(self, at_primal, at_state, at_adjoint, primal_work):
        """
        Computes in-place the total derivative of the objective function.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        at_adjoint : StateVector
            Current adjoint variables.
        primal_work : PrimalVector
            Temporary work vector of Primal type.
        """
        # first compute the objective partial
        primal_work.equals_objective_partial(at_primal, at_state)
        # construct the residual jacobian
        jacobian = dRdX(at_primal, at_state)
        # multiply the adjoint variables with the jacobian
        jacobian.T.product(at_adjoint, self)
        # add it to the objective partial
        self.plus(primal_work)

    def equals_lagrangian_total_gradient(self, at_primal, at_state, at_dual,
                                           at_adjoint, primal_work):
        """
        Computes in-place the total derivative of the Lagrangian.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        at_dual : DualVector
            Current lagrange multipliers.
        at_adjoint : StateVector
            Current adjoint variables.
        primal_work : PrimalVector
            Temporary work vector of Primal type.
        """
        # first compute the total derivative of the objective
        self.equals_total_gradient(at_primal, at_state, at_adjoint, primal_work)
        # initialize the linearized constraint jacobian
        cnstr_jac = dCdX(at_primal, at_state)
        # multiply the lagrange multipliers by the constraint jacobian
        cnstr_jac.T.product(at_dual, primal_work)
        # add it to the total objective derivative
        self.plus(primal_work)

class StateVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific
    to state vectors.
    """
    def equals_objective_partial(self, at_primal, at_state):
        """
        Computes in-place the partial derivative of the objective function with
        respect to state variables.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        self._memory.solver.eval_dFdU(
            at_primal._data, at_state._data, self._data
            )

    def equals_residual(self, at_primal, at_state):
        """
        Computes in-place the system residual vector.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        self._memory.solver.eval_residual(at_primal._data,
                                          at_state._data,
                                          self._data)

    def equals_primal_solution(self, at_primal):
        """
        Performs a non-linear system solution at the given primal point and
        stores the result in-place.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        """
        self._memory.solver.solve_nonlinear(at_primal._data, self._data)

    def equals_adjoint_solution(self, at_primal, at_state, state_work):
        """
        Computes in-place the adjoint variables for the objective function,
        linearized at the given primal and state points.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        state_work : StateVector
            Temporary work vector of State type.
        """
        jacobian = dRdU(at_primal, at_state)
        state_work.equals_objective_partial(at_primal, at_state)
        state_work.times(-1) # negative of the objective partial (-dF/dU)
        jacobian.T.solve(state_work, 1.e-12, self) # this is the adjoint solution now

class DualVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific
    to state vectors.
    """
    def convert(self, primal_vector):
        """
        Copy target state variables from the design space into dual space. Used
        only for IDF problems.

        Parameters
        ----------
        primal_vector : PrimalVector
            Source vector for target state variable data.
        """
        self._memory.solver.copy_targstate_to_dual(primal_vector._data, self._data)

    def equals_constraints(self, at_primal, at_state):
        """
        Evaluate all constraints at the given primal and state points, and
        store the result in-place.

        Parameters
        ----------
        at_primal : PrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        self._memory.solver.eval_ceq(at_primal._data,
                                     at_state._data,
                                     self._data)
