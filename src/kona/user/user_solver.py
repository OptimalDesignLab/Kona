import numpy as np

from kona.user import BaseVector

class UserSolver(object):
    """Base class for Kona objective functions, designed to be a template for
    any objective functions intended to be used for ``kona.Optimize()``.

    This class provides some standard mathematical functionality via NumPy
    arrays and operators. However, attributes of the derived classes can have
    different data types. In these cases, the user must redefine the
    mathematical operation methods for these non-standard data types.

    This solver wrapper is not initialized by Kona. The user must initialize it
    before handing it over to Kona's optimizer. Therefore the intialization
    implementation details are left up to the user entirely. Below is just an
    example used by Kona's own test problems.

    Parameters
    ----------
    num_design : int
        Design space size
    num_state : int, optional
        State space size.
    num_eq : int, optional
        Number of equality constraints
    num_ineq : int, optional
        Number of inequality constraints

    Attributes
    ----------
    num_design : int
        Size of the design space
    num_state : int
        Number of state variables
    num_eq : int
        Number of equality constraints
    num_ineq : int
        Number of inequality constraints
    """

    def __init__(self, num_design, num_state=0, num_eq=0, num_ineq=0):
        assert num_design > 0, \
            "Problem must have design variables!"
        self.num_design = num_design
        self.num_state = num_state
        self.num_eq = num_eq
        self.num_ineq = num_ineq

    def get_rank(self):
        """
        Rank of current process is needed purely for purposes of printing to
        screen
        """
        return 0

    def allocate_state(self, num_vecs):
        """
        Allocate the requested number of state-space BaseVectors and return
        them in a plain array.

        Parameters
        ----------
        num_vecs : int
            Number of state vectors requested.

        Returns
        -------
        list of BaseVector
            Stack of BaseVectors in the state space
        """
        return [BaseVector(self.num_state) for i in range(num_vecs)]

    def eval_obj(self, at_design, at_state):
        """
        Evaluate the objective function using the design variables stored at
        ``at_design`` and the state variables stored at ``at_state``.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.

        Returns
        -------
        tuple
            Result of the operation. Contains the objective value as the first
            element, and the number of preconditioner calls used as the second.
        """
        raise NotImplementedError

    def eval_residual(self, at_design, at_state, store_here):
        """
        Evaluate the governing equations (discretized PDE residual) at the
        design variables stored in ``at_design`` and the state variables stored
        in ``at_state``. Put the residual vector in ``store_here``.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        result : BaseVector
            Location where user should store the result.
        """
        store_here.data[:] = 0.

    def eval_eq_cnstr(self, at_design, at_state):
        """
        Evaluate the vector of equality constraints using the given design and
        state vectors.

        The constraints must have the form (c - c_target). In other words, the
        constraint value should be zero at feasibility.

        * For inequality constraints, the constraint value should be greater
          than zero when feasible.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.

        Returns
        -------
        result : numpy.ndarray
            Array of equality constraints.
        """
        return np.zeros(self.num_eq)

    def eval_ineq_cnstr(self, at_design, at_state):
        """
        Evaluate the vector of inequality constraints using the given design
        and state vectors.

        The constraints must have the form (c - c_target) > 0. In other words, the
        constraint value should be greater than zero when feasible.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.

        Returns
        -------
        result : numpy.ndarray
            Array of equality constraints.
        """
        return np.zeros(self.num_ineq)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the matrix-vector product for the design-jacobian of the PDE
        residual. The multiplying vector is ``in_vec`` and the result should be
        stored in ``out_vec``. The product should be evaluated at the given
        design and state vectors, ``at_design`` and ``at_state`` respectively.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial X} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : numpy.ndarray
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        out_vec.data[:] = 0.

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the matrix-vector product for the state-jacobian of the PDE
        residual. The multiplying vector is ``in_vec`` and the result should be
        stored in ``out_vec``. The product should be evaluated at the given
        design and state vectors, ``at_design`` and ``at_state`` respectively.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial U} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        out_vec.data[:] = 0.

    def multiply_dRdX_T(self, at_design, at_state, in_vec):
        """
        Evaluate the transposed matrix-vector product for the design-jacobian
        of the PDE residual. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial X}^T in_vec =
            out_vec

        .. note::

            Must always store a result even when it isn't implemented. Use a
            zero vector of length ``self.num_design`` for this purpose.

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : numpy.ndarray
            Location where user should store the result.
        """
        return np.zeros(self.num_design)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the transposed matrix-vector product for the state-jacobian
        of the PDE residual. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial U}^T in_vec =
            out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        out_vec.data[:] = 0.

    def factor_linear_system(self, at_design, at_state):
        """
        OPTIONAL: Build/factor the dR/dU matrix and its preconditioner at the
        given design and state vectors, ``at_design`` and ``at_state``. These
        matrices are then used to perform forward solves, adjoint solves and
        forward/transpose preconditioner applications.

        This routine is only used by matrix-based solvers where matrix
        factorizations are costly and should be done only once per optimization
        iteration. The optimization options dictionary must have the
        ``matrix_explicit`` key set to ``True``.

        .. note::

            If the user chooses to leverage this factorization, the
            (design, state) evaluation points should be ignored for
            preconditioner application, linear solve, and adjoint
            solve calls.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        """
        pass

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        """
        Apply the preconditioner to the vector at ``in_vec`` and
        store the result in ``out_vec``. If the preconditioner is nonlinear,
        evaluate the application using the design and state vectors provided
        in ``at_design`` and ``at_state``.

        .. note::

            If the solver uses ``factor_linear_system()``, ignore the
            (design, state) evaluation point and use the previously factored
            preconditioner.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.

        Returns
        -------
        int
            Number of preconditioner calls required for the operation.
        """
        out_vec.data[:] = in_vec.data[:]
        return 0

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        """
        Apply the transposed preconditioner to the vector at ``in_vec`` and
        store the result in ``out_vec``. If the preconditioner is nonlinear,
        evaluate the application using the design and state vectors provided
        in ``at_design`` and ``at_state``.

        .. note::

            If the solver uses ``factor_linear_system()``, ignore the
            (design, state) evaluation point and use the previously factored
            preconditioner.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.

        Returns
        -------
        int
            Number of preconditioner calls required for the operation.
        """
        out_vec.data[:] = in_vec.data[:]
        return 0

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        """
        Evaluate the matrix-vector product for the design-jacobian of the
        equality constraints. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C_{eq}(at_design, at_state)}{\partial X} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : numpy.ndarray
            Vector to be operated on.

        Returns
        -------
        numpy.ndarray
            Result of the product.
        """
        assert len(in_vec) == self.num_design, "Incorrect vector size!"
        return np.zeros(self.num_eq)

    def multiply_dCEQdU(self, at_design, at_state, in_vec):
        """
        Evaluate the matrix-vector product for the state-jacobian of the
        equality constraints. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C_{eq}(at_design, at_state)}{\partial U} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.

        Returns
        -------
        numpy.ndarray
            Result of the product
        """
        return np.zeros(self.num_eq)

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        """
        Evaluate the transposed matrix-vector product for the design-jacobian
        of the equality constraints. The multiplying vector is ``in_vec`` and
        the result should be stored in ``out_vec``. The product should be
        evaluated at the given design and state vectors, ``at_design`` and
        ``at_state`` respectively.

        .. math::

            \frac{\partial C_{eq}(at_design, at_state)}{\partial X}^T in_vec =
            out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : numpy.ndarray
            Vector to be operated on.

        Returns
        -------
        numpy.ndarray
            Result of the product.
        """
        assert len(in_vec) == self.num_eq, "Incorrect vector size!"
        return np.zeros(self.num_design)

    def multiply_dCEQdU_T(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the transposed matrix-vector product for the state-jacobian
        of the equality constraints. The multiplying vector is ``in_vec`` and
        the result should be stored in ``out_vec``. The product should be
        evaluated at the given design and state vectors, ``at_design`` and
        ``at_state`` respectively.

        .. math::

            \frac{\partial C_{eq}(at_design, at_state)}{\partial U}^T in_vec =
            out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : numpy.ndarray
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        assert len(in_vec) == self.num_eq, "Incorrect vector size!"
        out_vec.data[:] = 0.0

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        """
        Evaluate the matrix-vector product for the design-jacobian of the
        inequality constraints. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C_{ineq}(at_design, at_state)}{\partial X} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : numpy.ndarray
            Vector to be operated on.

        Returns
        -------
        numpy.ndarray
            Result of the product.
        """
        assert len(in_vec) == self.num_design, "Incorrect vector size!"
        return np.zeros(self.num_ineq)

    def multiply_dCINdU(self, at_design, at_state, in_vec):
        """
        Evaluate the matrix-vector product for the state-jacobian of the
        inequality constraints. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C_{eq}(at_design, at_state)}{\partial U} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.

        Returns
        -------
        numpy.ndarray
            Result of the product
        """
        return np.zeros(self.num_ineq)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        """
        Evaluate the transposed matrix-vector product for the design-jacobian
        of the inequality constraints. The multiplying vector is ``in_vec`` and
        the result should be stored in ``out_vec``. The product should be
        evaluated at the given design and state vectors, ``at_design`` and
        ``at_state`` respectively.

        .. math::

            \frac{\partial C_{ineq}(at_design, at_state)}{\partial X}^T in_vec =
            out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : numpy.ndarray
            Vector to be operated on.

        Returns
        -------
        numpy.ndarray
            Result of the product.
        """
        assert len(in_vec) == self.num_ineq, "Incorrect vector size!"
        return np.zeros(self.num_design)

    def multiply_dCINdU_T(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the transposed matrix-vector product for the state-jacobian
        of the inequality constraints. The multiplying vector is ``in_vec`` and
        the result should be stored in ``out_vec``. The product should be
        evaluated at the given design and state vectors, ``at_design`` and
        ``at_state`` respectively.

        .. math::

            \frac{\partial C_{ineq}(at_design, at_state)}{\partial U}^T in_vec =
            out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : numpy.ndarray
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        assert len(in_vec) == self.num_ineq, "Incorrect vector size!"
        out_vec.data[:] = 0.0

    def enforce_bounds(self, design_vector):
        """
        Evaluate the design vector element-wise. If a design variable violates
        a bound, set it to the bound value.

        Parameters
        ----------
        design_vector : BaseVector
            Design vector over which bounds will be enforced.
        """
        pass

    def eval_dFdX(self, at_design, at_state):
        """
        Evaluate the partial of the objective w.r.t. design variables at the
        design point stored in ``at_design`` and the state variables stored in
        ``at_state``. Store the result in ``store_here``.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.

        Returns
        ------
        numpy.ndarray
            Gradient vector.
        """
        raise NotImplementedError
        return np.zeros(self.num_design)

    def eval_dFdU(self, at_design, at_state, store_here):
        """
        Evaluate the partial of the objective w.r.t. state variable at the
        design point stored in ``at_design`` and the state variables stored in
        ``at_state``. Store the result in ``store_here``.

        .. note::

            If there are no state variables, a zero vector must be stored.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector
            Current state vector.
        store_here : BaseVector
            Location where user should store the result.
        """
        store_here.data[:] = 0.

    def init_design(self):
        """
        Initialize the first design point. Store the design vector at
        ``store_here``. The optimization will start from this point.

        .. note::

            This method must be implemented for any problem type.

        Returns
        ------
        numpy.ndarray
            Initial design vector.
        """
        raise NotImplementedError
        return np.zeros(self.num_design)

    def solve_nonlinear(self, at_design, result):
        """
        Compute the state variables at the given design point, ``at_design``.
        Store the resulting state variables in ``result``.

        For linear problems, this can be a simple linear system solution:

        .. math::

            \mathcal{K}(x)\mathbf{u} = \mathbf{F}(x)

        For nonlinear problems, this can involve Newton iterations:

        .. math::

            \frac{\partial R(x, u_{guess})}{\partual u} \Delta u =
            -R(x, u_{guess})

        If the solution fails to converge, the user should return a
        negative integer in order to help Kona intelligently backtrack in the
        optimization.

        Similarly, in the case of correct convergence, the user is encouraged
        to return the number of preconditioner calls it took to solve the
        nonlinear system. Kona uses this information to track the
        computational cost of the optimization. If the number of preconditioner
        calls is not available, return a 1 (one).

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        result : BaseVector
            Location where user should store the result.

        Returns
        -------
        int
            Number of preconditioner calls required for the solution.
        """
        converged = True
        cost = 0
        result.data[:] = 0.
        if converged:
            # You can return the number of preconditioner calls here.
            # This is used by Kona to track computational cost.
            return cost
        else:
            # Must return negative cost to Kona when your system solve fails to
            # converge. This is important because it helps Kona determine when
            # it needs to back-track on the optimization.
            return -cost

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        """
        Solve the linear system defined by the state-jacobian of the PDE
        residual, to the specified absolute tolerance ``tol``.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial U} result = rhs_vec

        The jacobian should be evaluated at the given (design, state) point,
        ``at_design`` and ``at_state``.

        Store the solution in ``result``.

        .. note::

            If the solver uses ``factor_linear_system()``, ignore the
            ``at_design`` evaluation point and use the previously factored
            preconditioner.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector-line
            Current state vector.
        rhs_vec : BaseVector
            Right hand side vector.
        rel_tol : float
            Tolerance that the linear system should be solved to.
        result : BaseVector
            Location where user should store the result.

        Returns
        -------
        int
            Number of preconditioner calls required for the solution.
        """
        if rhs_vec.data == 0.:
            result.data[:] = 0.
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, tol, result):
        """
        Solve the linear system defined by the transposed state-jacobian of the
        PDE residual, to the specified absolute tolerance ``tol``.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial U}^T result =
            rhs_vec

        The jacobian should be evaluated at the given (design, state) point,
        ``at_design`` and ``at_state``.

        Store the solution in ``result``.

        .. note::

            If the solver uses ``factor_linear_system()``, ignore the
            ``at_design`` evaluation point and use the previously factored
            preconditioner.

        Parameters
        ----------
        at_design : numpy.ndarray
            Current design vector.
        at_state : BaseVector-line
            Current state vector.
        rhs_vec : BaseVector
            Right hand side vector.
        rel_tol : float
            Tolerance that the linear system should be solved to.
        result : BaseVector
            Location where user should store the result.

        Returns
        -------
        int
            Number of preconditioner calls required for the solution.
        """
        if rhs_vec.data == 0.:
            result.data[:] = 0.
        return 0

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        """
        Kona will evaluate this method at every outer/nonlinear optimization
        iteration. It can be used to print out useful information to monitor
        the process, or to save design points of the intermediate iterations.

        The current design vector, current state vector and current adjoint
        vector have been made available to the user via the arguments.

        Parameters
        ----------
        num_iter : int
            Current outer/nonlinear iteration number.
        curr_design : numpy.ndarray
            Current design point.
        curr_state : BaseVector
            Current state variables.
        curr_adj : BaseVector
            Currently adjoint variables for the Lagrangian.
        curr_eq : numpy.ndarray
            Current Lagrange multipliers for equality constraints.
        curr_ineq: numpy.ndarray
            Current Lagrange multipliers for inequality constraints.
        curr_slack : numpy.ndarray
            Current slack variables associated with inequality constraints.

        Returns
        -------
        string, optional
            A string that that Kona will write into its info file.
        """
        return ""

class UserSolverIDF(UserSolver):
    """
    A modified base class for multidisciplinary problems that adopt the
    IDF (individual discipline feasible) formulation for disciplinary
    coupling.

    Parameters
    ----------
    num_design : int
        Size of the design space, NOT including target state variables
    num_state : int
        Number of state variables
    num_idf : int
        Number of IDF constraints
    num_eq : int, optional
        Number of equality constraints outside of IDF
    num_ineq : int, optional
        Number of inequality constraints

    Attributes
    ----------
    num_real_design : int
        Size of the design space, NOT including target state variables
    num_design : int
        Number of TOTAL design variables, including target state variables
    num_state : int
        Number of state variables
    num_real_ceq : int
        Number of equality constraints, NOT including IDF constraints
    num_eq : int
        Number of TOTAL equality constraints, including IDF constraints
    num_ineq : int
        Number of inequality constraints
    """

    def __init__(self, num_design, num_state, num_idf, num_eq=0, num_ineq=0):
        assert num_design > 0, \
            "Problem must have design variables!"
        assert num_idf >= 0, \
            "Must have at least one IDF constraint!"
        assert num_state >= num_idf, \
            "Cannot have more IDF constraints than state variables!"
        self.num_real_design = num_design
        self.num_idf = num_idf
        self.num_real_ceq = num_eq
        super(UserSolverIDF, self).__init__(
            self.num_real_design + num_idf,
            num_state,
            self.num_real_ceq + num_idf,
            num_ineq)

    def restrict_design(self, opType, target):
        """
        If operation type is 0 (``type == 0``), set the target state variables
        to zero.

        If operation type is 1 (``type == 1``), set the real design variables
        to zero.

        Parameters
        ----------
        opType : int
            Operation type flag.
        vec : numpy.ndarray
            Design vector to be operated on.
        """
        if opType == 0:
            target[self.num_real_design:] = 0.
        elif opType == 1:
            target[:self.num_real_design] = 0.
        else:
            raise ValueError('Unexpected type in restrict_design()!')

    def copy_dual_to_targstate(self, take_from, copy_to):
        """
        Take the target state variables from dual storage and put them into
        design storage. Also set the real design variables to zero.

        Parameters
        ----------
        take_from : numpy.ndarray
            Vector from where target state variables should be taken.
        copy_to : numpy.ndarray
            Vector to which target state variables should be copied.
        """
        copy_to[:self.num_real_design] = 0.
        copy_to[self.num_real_design:] = take_from[self.num_real_ceq:]

    def copy_targstate_to_dual(self, take_from, copy_to):
        """
        Take the target state variables from design storage and put them into
        dual storage.

        Parameters
        ----------
        take_from : numpy.ndarray
            Vector from where target state variables should be taken.
        copy_to : numpy.ndarray
            Vector to which target state variables should be copied.
        """
        copy_to[self.num_real_ceq:] = take_from[self.num_real_design:]
