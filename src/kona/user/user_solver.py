from base_vectors import BaseAllocator

class UserSolver(object):
    """Base class for Kona objective functions, designed to be a template for
    any objective functions intended to be used for ``kona.Optimize()``.

    This class provides some standard mathematical functionality via NumPy
    arrays and operators. However, attributes of the derived classes can have
    different data types. In these cases, the user must redefine the
    mathematical operation methods for these non-standard data types.

    Parameters
    ----------
    num_primal : int, optional
        Primal space size.
    num_state : int, optional
        State space size.
    num_ceq : int, optional
        Dual space size.
    allocator : BaseAllocator, optional
        Allocator object that produces the abstracted vector data containers.

    Attributes
    ----------
    num_design : int
        Size of the design space
    num_state : int, optional
        Number of state variables
    num_ceq : int, optional
        Size of the equality constraint residual
    allocator : BaseAllocator
        Object that allocates BaseVector instances when Kona asks for it.
    """

    def __init__(self, num_primal=0, num_state=0, num_ceq=0, allocator=None):
        if allocator is None:
            self.allocator = BaseAllocator(num_primal, num_state, num_ceq)
        else:
            self.allocator = allocator
        self.num_primal = self.allocator.num_primal
        self.num_state = self.allocator.num_state
        self.num_dual = self.allocator.num_dual

    def get_rank(self):
        """
        Rank of current process is needed purely for purposes of printing to
        screen
        """
        return 0

    def eval_obj(self, at_design, at_state):
        """
        Evaluate the objective function using the design variables stored at
        ``at_design`` and the state variables stored at ``at_state``.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : BaseVector
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
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        result : BaseVector
            Location where user should store the result.
        """
        pass

    def eval_constraints(self, at_design, at_state, store_here):
        """
        Evaluate the vector of constraints using the design variables stored at
        ``at_design`` and the state variables stored at ``at_state``. Store the
        constraint residual at ``store_here``.

        The constraints must have the form (c - c_target). In other words:

        * For equality constraints, the constraint value should be zero at
          feasibility.

        * For inequality constraints, the constraint value should be greater
          than zero when feasible.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        result : BaseVector
            Location where user should store the result.
        """
        pass

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
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        pass

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
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        pass

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the transposed matrix-vector product for the design-jacobian
        of the PDE residual. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial X}^T in_vec = out_vec

        .. note::

            Must always store a result even when it isn't implemented. Use a
            zero vector of length ``self.num_design`` for this purpose.

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        out_vec.data[:] = 0.0

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the transposed matrix-vector product for the state-jacobian
        of the PDE residual. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial U}^T in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        pass

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
        at_design : BaseVector
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
        at_design : BaseVector
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
        at_design : BaseVector
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

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the matrix-vector product for the design-jacobian of the
        constraint vector. The multiplying vector is ``in_vec`` and the result
        should be stored in ``out_vec``. The product should be evaluated at the
        given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C(at_design, at_state)}{\partial X} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        pass

    def multiply_dCdU(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the matrix-vector product for the state-jacobian of the
        constraint vector. The multiplying vector is ``in_vec`` and the result
        should be stored in ``out_vec``. The product should be evaluated at the
        given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C(at_design, at_state)}{\partial U} in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        pass

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the transposed matrix-vector product for the design-jacobian
        of the constraint vector. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C(at_design, at_state)}{\partial X}^T in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        out_vec.data[:] = 0.

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        """
        Evaluate the transposed matrix-vector product for the state-jacobian
        of the constraint vector. The multiplying vector is ``in_vec`` and the
        result should be stored in ``out_vec``. The product should be evaluated
        at the given design and state vectors, ``at_design`` and ``at_state``
        respectively.

        .. math::

            \frac{\partial C(at_design, at_state)}{\partial U}^T in_vec = out_vec

        .. note::

            This jacobian is a partial. No total derivatives, gradients or
            jacobians should be evaluated by any UserSolver implementation.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        in_vec : BaseVector
            Vector to be operated on.
        out_vec : BaseVector
            Location where user should store the result.
        """
        pass

    def apply_active_set(self, at_constraints, in_vec, out_vec):
        """
        This is a method used only for inequality constrained problems.

        The active-set "matrix" is mathematically defined as a modified
        identity matrix with matching dimensions to the constraint vector.
        The diagonal entries of this matrix corresponding to inactive
        inequality constraints are set to zero.

        The user solver should parse the data in the given constraint vector,
        ``at_constraints``, evaluate feasibility, and then zero out the
        appropriate values for the given input vector, ``in_vec``. The result
        of the operation should be stored in ``out_vec``.

        .. note::

            The user should only evaluate feasibility for inequality
            constraints. No work should be done regarding any equality
            constraints.

        Parameters
        ----------
        at_constraints : BaseVector
        in_vec : BaseVector
        out_vec : BaseVector
        """
        out_vec.data[:] = in_vec.data[:]

    def eval_dFdX(self, at_design, at_state, store_here):
        """
        Evaluate the partial of the objective w.r.t. design variables at the
        design point stored in ``at_design`` and the state variables stored in
        ``at_state``. Store the result in ``store_here``.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        store_here : BaseVector
            Location where user should store the result.
        """
        raise NotImplementedError

    def eval_dFdU(self, at_design, at_state, store_here):
        """
        Evaluate the partial of the objective w.r.t. state variable at the
        design point stored in ``at_design`` and the state variables stored in
        ``at_state``. Store the result in ``store_here``.

        .. note::

            If there are no state variables, a zero vector must be stored.

        Parameters
        ----------
        at_design : BaseVector
            Current design vector.
        at_state : BaseVector
            Current state vector.
        store_here : BaseVector
            Location where user should store the result.
        """
        store_here.data[:] = 0.

    def init_design(self, store_here):
        """
        Initialize the first design point. Store the design vector at
        ``store_here``. The optimization will start from this point.

        .. note::

            This method must be implemented for any problem type.
        """
        raise NotImplementedError

    def solve_nonlinear(self, at_design, result):
        """
        Compute the state variables at the given design point, ``at_design``.
        Store the resulting state variables in ``result``.

        For linear problems, this can be a simple linear system solution:

        .. math:: \mathcal{K}(x)\mathbf{u} = \mathbf{F}(x)

        For nonlinear problems, this can involve Newton iterations:

        .. math:: \frac{\partial R(x, u_{guess})}{\partual u} \Delta u = -R(x, u_{guess})

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
        at_design : BaseVector
            Current design vector.
        result : BaseVector
            Location where user should store the result.

        Returns
        -------
        int
            Number of preconditioner calls required for the solution.
        """
        converged = True
        cost = 1
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
        at_design : BaseVector
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
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, tol, result):
        """
        Solve the linear system defined by the transposed state-jacobian of the
        PDE residual, to the specified absolute tolerance ``tol``.

        .. math::

            \frac{\partial R(at_design, at_state)}{\partial U}^T result = rhs_vec

        The jacobian should be evaluated at the given (design, state) point,
        ``at_design`` and ``at_state``.

        Store the solution in ``result``.

        .. note::

            If the solver uses ``factor_linear_system()``, ignore the
            ``at_design`` evaluation point and use the previously factored
            preconditioner.

        Parameters
        ----------
        at_design : BaseVector
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
        return 0

    def current_solution(self, curr_design, curr_state, curr_adj,
                         curr_dual, num_iter):
        """
        Kona will evaluate this method at every outer optimization iteration.
        It can be used to print out useful information to monitor the process,
        or to save design points of the intermediate iterations.

        The current design vector, current state vector and current adjoint
        vector have been made available to the user via the arguments.

        Parameters
        ----------
        curr_design : BaseVector
            Current design point.
        curr_state : BaseVector
            Current state variables.
        curr_adj : BaseVector
            Currently adjoint variables for the objective.
        curr_dual : BaseVector
            Current dual vector in storage. (This might be unnecessary!)
        num_iter : int
            Current outer iteration number.
        """
        self.curr_design = curr_design.data
        self.num_iter = num_iter

        if curr_state is not None:
            self.curr_state = curr_state.data

        if curr_adj is not None:
            self.curr_adj = curr_adj.data

        if curr_dual is not None:
            self.curr_dual = curr_dual.data

class UserSolverIDF(UserSolver):
    """
    A modified base class for multidisciplinary problems that adopt the
    IDF (individual discipline feasible) formulation for disciplinary
    coupling.

    Parameters
    ----------
    num_real_design : int
        Size of the design space, NOT including target state variables
    num_state : int
        Number of state variables
    num_real_ceq : int (optional)
        Number of equality constraints, NOT including IDF constraints

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
    num_ceq : int
        Number of TOTAL equality constraints, including IDF constraints
    """

    def __init__(self, num_design, num_state, num_real_ceq, allocator=None):
        self.num_real_design = num_design
        self.num_real_ceq = num_real_ceq
        super(UserSolverIDF, self).__init__(
            self.num_real_design + num_state,
            num_state,
            self.num_real_ceq + num_state,
            allocator)

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
            target.data[self.num_real_design:] = 0.
        elif opType == 1:
            target.data[:self.num_real_design] = 0.
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
        copy_to.data[:self.num_real_design] = 0.
        copy_to.data[self.num_real_design:] = take_from.data[self.num_real_ceq:]

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
        copy_to.data[self.num_real_ceq:] = take_from.data[self.num_real_design:]
