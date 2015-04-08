from numpy import sqrt

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
            raise TypeError('KonaVector() >> ' + \
                            'Vector type mismatch. Must be %s' % type(self))

    def equals(self, rhs): # the = operator cannot be overloaded
        """
        Used as the assignment operator.

        If RHS is a scalar, all vector elements are set to the scalar value.

        If RHS is a vector, the two vectors are set equal.

        Parameters
        ----------
        rhs : float or KonaVector derivative
            Right hand side term for assignment.
        """
        if isinstance(rhs, float):
            self._data.equals_value(rhs)
        else:
            self._check_type(rhs)
            self._data.equals_vector(rhs._data)

    def plus(self, vector): # this is the += operator
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

    def minus(self, vector): # this is the -= operator
        """
        Used as the subtraction operator.

        Subtracts the incoming vector from the current vector in place.

        Parameters
        ----------
        vector : KonaVector derivative
            Vector to be subtracted.
        """
        self._check_type(vector)
        self._data.times(-1.)
        self._data.plus(vector._data)
        self._data.times(-1.)

    def times(self, value): # this is the *= operator
        """
        Used as the multiplication operator.

        Multiplies the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        if not isinstance(value, float):
            self._data.times(value)
        else:
            raise TypeError('KonaVector() >> ' + \
                            'Argument must be a float.')

    def divide_by(self, val): # this is the /= operator
        """
        Used as the multiplication operator.

        Multiplies the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        self.times(1./val)

    def equals_ax_p_by(self, a, x, b, y): # this performs self = a*x + b*y
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
        self._check_type(x)
        self._check_type(y)
        self._data.equals_ax_p_by(a, x._data, b, y._data)

    def inner(self, vector):
        """
        Computes an inner product with another vector.

        Returns
        -------
        float : Inner product.
        """
        self.check_type(vector)
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
        if prod < 0:
            raise ValueError('KonaVector.norm2 >> ' + \
                             'Inner product is negative!')
        else:
            return sqrt(prod)

class PrimalVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific
    to design vectors.
    """
    def restrict_target_state(self):
        self._memory.solver.allocator.restrict_design(0, self._data)

    def restrict_real_design(self):
        self._memory.solver.allocator.restrict_design(1, self._data)

    def convert(self, dual_vector):
        self._memory.solver.allocator.copy_dual_to_targstate(dual_vector._data,
                                                     self._data)

    def equals_init_design(self):
        self._memory.solver.init_design(self._data)

    def equals_objective_gradient(self, at_design, at_state):
        self._memory.solver.eval_obj_d(at_design._data,
                                       at_state._data,
                                       self._data)

    def equals_reduced_gradient(self, at_design, at_state, at_adjoint, work):
        pass

    def equals_lagrangian_reduced_grad(self, at_design, atstate, at_dual,
                                       at_adjoint, work):
        pass

class StateVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific
    to state vectors.
    """
    def equals_objective_partial(self, at_design, at_state):
        self._memory.solver.eval_obj_s(
            at_design._data, at_state._data, self._data
            )

    def equals_PDE_residual(self, at_design, at_state):
        self._memory.solver.eval_residual(at_design._data,
                                          at_state._data,
                                          self._data)

    def equals_primal_solution(self, at_design):
        self._memory.solver.solve_system(at_design._data, self._data)

class DualVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific
    to state vectors.
    """

    def convert(self, design):
        self._memory.solver.copy_targstate_to_dual(design._data, self._data)

    def equals_constraints(self, at_design, at_state):
        self._memory.solver.eval_ceq(at_design._data,
                                     at_state._data,
                                     self._data)
