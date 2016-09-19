
class KonaVector(object):
    """
    An abstract vector class connected to the Kona memory, containing a
    common set of algebraic member functions. Allows Kona to operate on
    data spaces allocated by the user.

    Parameters
    ----------
    memory_obj : KonaMemory
        Pointer to the Kona user memory.
    user_vector : BaseVector
        User defined vector object that contains data and operations on data.

    Attributes
    ----------
    _memory : UserMemory
        Pointer to the Kona user memory.
    base : BaseVector
        User defined vector object that contains data and operations on data.
    """

    def __init__(self, memory_obj, user_vector=None):
        self._memory = memory_obj
        self.base = user_vector

    def __del__(self):
        self._memory.push_vector(type(self), self.base)

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
        val : float or KonaVector
            Right hand side term for assignment.
        """
        if isinstance(val,
                      (float, np.float32, np.float64, int, np.int32, np.int64)):
            self.base.equals_value(val)
        else:
            assert isinstance(val, type(self))
            self.base.equals_vector(val.base)

    def plus(self, vector):
        """
        Used as the addition operator.

        Adds the incoming vector to the current vector in place.

        Parameters
        ----------
        vector : KonaVector
            Vector to be added.
        """
        assert isinstance(vector, type(self))
        self.base.plus(vector.base)

    def minus(self, vector):
        """
        Used as the subtraction operator.

        Subtracts the incoming vector from the current vector in place.

        Parameters
        ----------
        vector : KonaVector
            Vector to be subtracted.
        """
        if vector == self: # special case...
            self.equals(0)

        assert isinstance(vector, type(self))
        self.base.times_scalar(-1.)
        self.base.plus(vector.base)
        self.base.times_scalar(-1.)

    def times(self, factor):
        """
        Used as the multiplication operator.

        Can multiply both by scalars or element-wise by vectors.

        Parameters
        ----------
        factor : float or KonaVector
            Scalar or vector-valued multiplication factor.
        """
        if isinstance(factor,
                      (float, np.float32, np.float64, int, np.int32, np.int64)):
            self.base.times_scalar(factor)
        else:
            assert isinstance(factor, type(self))
            self.base.times_vector(factor.base)

    def divide_by(self, val):
        """
        Used as the division operator.

        Divides the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        if val == 0.0:
            raise ValueError('Divide by zero!')
        self.times(1./val)

    def equals_ax_p_by(self, a, X, b, Y):
        """
        Performs the scaled summation operation, ``a*X + b*Y``, between two
        vectors, and stores the result in place.

        Parameters
        ----------
        a : float
            Scalar coefficient for ``X``.
        X : KonaVector
            Vector for the operation.
        b : float
            Scalar coefficient for ``Y``.
        Y : KonaVector
            Vector for the operation.
        """
        assert isinstance(X, type(self))
        assert isinstance(Y, type(self))
        self.base.equals_ax_p_by(a, X.base, b, Y.base)

    def exp(self, vector):
        """
        Performs an element-wise exponential operation on the given vector
        and stores the result in-place.

        Parameters
        ----------
        vector : KonaVector
            Vector for the operation.
        """
        assert isinstance(vector, type(self))
        self.base.exp(vector.base)

    def log(self, vector):
        """
        Performs an element-wise natural log operation on the given vector
        and stores the result in-place.

        Parameters
        ----------
        vector : KonaVector
            Vector for the operation.
        """
        assert isinstance(vector, type(self))
        self.base.log(vector.base)

    def pow(self, power):
        """
        Performs an element-wise power operation in-place.

        Parameters
        ----------
        power : float
        """
        self.base.pow(power)

    def inner(self, vector):
        """
        Computes an inner product with another vector.

        Returns
        -------
        float
            Inner product.
        """
        assert isinstance(vector, type(self))
        return self.base.inner(vector.base)

    @property
    def norm2(self): # this takes the L2 norm of the vector
        """
        Computes the L2 (Euclidian) norm of the vector.

        Returns
        -------
        float
            L2 norm.
        """
        prod = self.inner(self)
        return np.sqrt(prod)

    @property
    def infty(self):
        """
        Computes the infinity norm of the vector

        Returns
        -------
        float
            Infinity norm.
        """
        return self.base.infty

class DesignVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific
    to design vectors.
    """

    def __init__(self, memory_obj, user_vector=None):
        super(DesignVector, self).__init__(memory_obj, user_vector)
        self.lb = self._memory.design_lb
        self.ub = self._memory.design_ub

    def restrict_to_design(self):
        """
        Set target state variables to zero, leaving design variables untouched.

        Used only for IDF problems.
        """
        self._memory.solver.restrict_design(0, self.base.data)

    def restrict_to_target(self):
        """
        Set design variables to zero, leaving target state variables untouched.

        Used only for IDF problems.
        """
        self._memory.solver.restrict_design(1, self.base.data)

    def convert(self, dual_vector):
        """
        Copy target state variables from the dual space into the design space.

        Used only for IDF problems.

        Parameters
        ----------
        dual_vector : DualVectorEQ
            Source vector for target state variable data.
        """
        assert isinstance(dual_vector, DualVectorEQ), \
            "Invalid dual vector: must be DualVectorEQ!"
        self._memory.solver.copy_dual_to_targstate(
            dual_vector.base.data, self.base.data)

    def enforce_bounds(self):
        """
        Element-wise enforcement of design bounds.
        """
        if self.lb is not None:
            for i in xrange(len(self.base.data)):
                if self.base.data[i] < self.lb:
                    self.base.data[i] = self.lb
        if self.ub is not None:
            for i in xrange(len(self.base.data)):
                if self.base.data[i] > self.ub:
                    self.base.data[i] = self.ub

    def equals_init_design(self):
        """
        Sets this vector equal to the initial design point.
        """
        self.base.data[:] = self._memory.solver.init_design()

    def equals_objective_partial(self, at_primal, at_state):
        """
        Computes in-place the partial derivative of the objective function with
        respect to design variables.

        Parameters
        ----------
        at_primal : DesignVector or CompositePrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
        elif isinstance(at_primal, DesignVector):
            at_design = at_primal
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        assert isinstance(at_state, StateVector), \
            "Invalid state vector type: must be StateVector!"
        self.base.data[:] = self._memory.solver.eval_dFdX(
            at_design.base.data, at_state.base)

    def equals_total_gradient(self, at_primal, at_state, at_adjoint):
        """
        Computes in-place the total derivative of the objective function.

        Parameters
        ----------
        at_primal : DesignVector or CompositePrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        at_adjoint : StateVector
            Current adjoint variables.
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
        elif isinstance(at_primal, DesignVector):
            at_design = at_primal
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        assert isinstance(at_state, StateVector), \
            "Invalid state vector type: must be StateVector!"
        assert isinstance(at_adjoint, StateVector), \
            "Invalid adjoint vector type: must be StateVector!"

        # first compute the objective partial
        self.equals_objective_partial(at_design, at_state)
        # multiply the adjoint variables with the jacobian
        self.base.data[:] += self._memory.solver.multiply_dRdX_T(
            at_design.base.data, at_state.base, at_adjoint.base)

    def equals_lagrangian_total_gradient(self, at_primal, at_state, at_dual,
                                         at_adjoint):
        """
        Computes in-place the total derivative of the Lagrangian.

        Parameters
        ----------
        at_primal : DesignVector or CompositePrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        at_dual : DualVectorEQ, DualVectorINEQ or CompositeDualVector
            Current lagrange multipliers.
        at_adjoint : StateVector
            Current adjoint variables for the Lagrangian (rhs = - dL/dU)
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
            if isinstance(at_dual, CompositeDualVector):
                at_dual_eq = at_dual.eq
                at_dual_ineq = at_dual.ineq
            elif isinstance(at_dual, DualVectorINEQ):
                at_dual_eq = None
                at_dual_ineq = at_dual
            else:
                raise AssertionError(
                    "Invalid dual vector type: " +
                    "must be DualVectorINEQ or CompositeDualVector!")
        elif isinstance(at_primal, DesignVector):
            assert isinstance(at_dual, DualVectorEQ), \
                "Invalid dual vector type: must be DualVectorEQ!"
            at_design = at_primal
            at_dual_eq = at_dual
            at_dual_ineq = None
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        assert isinstance(at_state, StateVector), \
            "Invalid state vector type: must be StateVector!"
        assert isinstance(at_adjoint, StateVector), \
            "Invalid adjoint vector type: must be StateVector!"
        # first compute the total derivative of the objective
        self.equals_total_gradient(at_primal, at_state, at_adjoint)
        # add the lagrange multiplier products for equality constraints
        if at_dual_eq is not None:
            self.base.data[:] += self._memory.solver.multiply_dCEQdX_T(
                at_design.base.data, at_state.base, at_dual_eq.base.data)
        # now do it for inequality constraints
        if at_dual_ineq is not None:
            self.base.data[:] += self._memory.solver.multiply_dCINdX_T(
                at_design.base.data, at_state.base, at_dual_ineq.base.data)

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
        at_primal : DesignVector or CompositePrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
        elif isinstance(at_primal, DesignVector):
            at_design = at_primal
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        assert isinstance(at_state, StateVector), \
            "Invalid state vector type: must be StateVector!"
        self._memory.solver.eval_dFdU(
            at_design.base.data, at_state.base, self.base)

    def equals_residual(self, at_primal, at_state):
        """
        Computes in-place the system residual vector.

        Parameters
        ----------
        at_primal : DesignVector or CompositePrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
        elif isinstance(at_primal, DesignVector):
            at_design = at_primal
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        assert isinstance(at_state, StateVector), \
            "Invalid state vector type: must be StateVector!"
        self._memory.solver.eval_residual(
            at_design.base.data, at_state.base, self.base)

    def equals_primal_solution(self, at_primal):
        """
        Performs a non-linear system solution at the given primal point and
        stores the result in-place.

        Parameters
        ----------
        at_primal : DesignVector
            Current primal point.
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
        elif isinstance(at_primal, DesignVector):
            at_design = at_primal
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        cost = self._memory.solver.solve_nonlinear(
            at_design.base.data, self.base)
        self._memory.cost += abs(cost)
        if cost < 0:
            return False
        else:
            return True

    def equals_adjoint_solution(self, at_primal, at_state, state_work):
        """
        Computes in-place the adjoint variables for the objective function,
        linearized at the given primal and state points.

        Parameters
        ----------
        at_primal : DesignVector
            Current primal point.
        at_state : StateVector
            Current state point.
        state_work : StateVector
            Temporary work vector of State type.
        """
        assert isinstance(state_work, StateVector), \
            "Invalid work vector type: must be StateVector!"
        state_work.equals_objective_partial(at_primal, at_state)
        state_work.times(-1) # negative of the objective partial (-dF/dU)
        dRdU(at_primal, at_state).T.solve(state_work, self)

    def equals_lagrangian_adjoint(self, at_kkt, at_state, state_work):
        """
        Computes in-place the adjoint variables for the augmented Lagrangian,
        linearized at the given KKT vector and state points.

        Parameters
        ----------
        at_kkt : ReducedKKTVector
            Current KKT point.
        at_state : StateVector
            Current state point.
        adj_work : StateVector
            Temporary work vector of State type.
        state_work : StateVector
            Temporary work vector of State type.
        """
        # assemble the right hand side for the Lagrangian adjoint solve
        assert isinstance(at_kkt, ReducedKKTVector), \
            "Invalid KKT vector: must be ReducedKKTVector!"
        at_primal = at_kkt.primal
        at_dual = at_kkt.dual
        state_work.equals_objective_partial(at_primal, at_state)
        dCdU(at_primal, at_state).T.product(at_dual, self, state_work)
        # perform adjoint solution
        state_work.times(-1.)
        dRdU(at_primal, at_state).T.solve(state_work, self)

class DualVector(KonaVector):
    pass

class DualVectorEQ(DualVector):

    def convert(self, primal_vector):
        """
        Copy target state variables from the design space into dual space. Used
        only for IDF problems.

        Parameters
        ----------
        primal_vector : DesignVector
            Source vector for target state variable data.
        """
        assert isinstance(primal_vector, DesignVector), \
            "Invalid primal vector: must be DesignVector!"
        self._memory.solver.copy_targstate_to_dual(
            primal_vector.base.data, self.base.data)

    def equals_constraints(self, at_primal, at_state):
        """
        Evaluate all equality constraints at the given primal and state points,
        and store the result in-place.

        Parameters
        ----------
        at_primal : DesignVector or CompositePrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
        elif isinstance(at_primal, DesignVector):
            at_design = at_primal
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        assert isinstance(at_state, StateVector), \
            "Invalid state vector type: must be StateVector!"
        self.base.data[:] = self._memory.solver.eval_eq_cnstr(
            at_design.base.data, at_state.base)

class DualVectorINEQ(DualVector):

    def equals_constraints(self, at_primal, at_state):
        """
        Evaluate all in-equality constraints at the given primal and state
        points, and store the result in-place.

        Parameters
        ----------
        at_primal : DesignVector or CompositePrimalVector
            Current primal point.
        at_state : StateVector
            Current state point.
        """
        if isinstance(at_primal, CompositePrimalVector):
            at_design = at_primal.design
        elif isinstance(at_primal, DesignVector):
            at_design = at_primal
        else:
            raise AssertionError(
                "Invalid primal vector type: " +
                "must be DesignVector or CompositePrimalVector!")
        assert isinstance(at_state, StateVector), \
            "Invalid state vector type: must be StateVector!"
        self.base.data[:] = self._memory.solver.eval_ineq_cnstr(
            at_design.base.data, at_state.base)


# package imports at the bottom to prevent import errors
import numpy as np
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.common import *