from base_vectors import BaseAllocator

class UserSolver(object):
    """Base class for Kona objective functions, designed to be a template for
    any objective functions intended to be used for ``kona.Optimize()``.

    This class provides some standard mathematical functionality via NumPy
    arrays and operators. However, attributes of the derived classes can have
    different data types. In these cases, the user must redefine the
    mathematical operation methods for these non-standard data types.

    Attributes
    ----------
    num_design : int
        Size of the design space
    num_state : int (optional)
        Number of state variables
    num_ceq : int (optional)
        Size of the equality constraint residual
    allocator : BaseAllocator-like
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
        Rank of current process is needed purely for purposes of printing to screen
        """
        return 0

    def eval_obj(self, at_design, at_state):
        """
        Evaluate the objective function using the design variables stored at
        ``at_design`` and the state variables stored at ``at_state``.

        This function should return a tuple containing the objective function
        value as the first element, and the number of preconditioner calls as
        the second.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.

        Returns
        -------
        tuple : Result of the operation. Contains the objective value as the
        first element, and the number of preconditioner calls used as the
        second.
        """
        raise NotImplementedError

    def eval_residual(self, at_design, at_state, store_here):
        """
        Evaluate the linearized system (PDE) using the design point stored in
        ``at_design`` and the state variables stored in ``at_state``. Put the
        residual vector in ``store_here``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        result : BaseVector-like
            Location where user should store the result.
        """
        pass

    def eval_ceq(self, at_design, at_state, store_here):
        """
        Evaluate the vector of equality constraints using the design variables
        stored at `at_design`` and the state variables stored at ``at_state``.
        Store the constraint residual at ``store_here``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        result : BaseVector-like
            Location where user should store the result.
        """
        pass

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Update the linearized system jacobian using the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the design component of the system jacobian with the vector
          stored in ``in_vec``.

        * Store the result in ``out_vec``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        pass

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Update the linearized system jacobian using the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the state component of the system jacobian with the vector
          stored in ``in_vec``.

        * Store the result in ``out_vec``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        pass

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Update the linearized system jacobian using the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the transposed design component of the system jacobian with
          the vector stored in ``in_vec``.

        * Store the result in ``out_vec``.

        .. note::

            Must always store a result even when it isn't implemented. Use a
            zero vector of length ``self.num_design`` for this purpose.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        result.data[:] = 0.0

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Update the linearized system jacobian using the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the transposed state component of the system jacobian with
          the vector stored in ``in_vec``.

        * Store the result in ``out_vec``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        pass

    def build_precond(self):
        """
        OPTIONAL: Build the system preconditioner.
        """
        pass

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        """
        OPTIONAL: Apply the preconditioner to the vector at ``in_vec`` and
        store the result in ``out_vec``. If the preconditioner has to be
        linearized, use the design and state vectors provided in ``at_design``
        and ``at_state``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the operation.
        """
        return 0

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        """
        OPTIONAL: Apply the transposed preconditioner to the vector at
        ``in_vec`` and store the result in ``out_vec``. If the preconditioner
        has to be linearized, use the design and state vectors provided in
        ``at_design`` and ``at_state``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the operation.
        """
        return 0

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Linearize the constraint jacobian about the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the design component of the constraint jacobian with
          the vector stored in ``in_vec``.

        * Store the result in ``out_vec``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        pass

    def multiply_dCdU(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Linearize the constraint jacobian about the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the state component of the constraint jacobian with
          the vector stored in ``in_vec``.

        * Store the result in ``out_vec``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        pass

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Linearize the constraint jacobian about the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the transposed design component of the constraint jacobian
          with the vector stored in ``in_vec``.

        * Store the result in ``out_vec``.

        .. note::

            Must always store a result even when it isn't implemented. Use a
            zero vector of length ``self.num_design`` for this purpose.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        out_vec.data[:] = 0.

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        """
        Perform the following tasks:

        * Linearize the constraint jacobian about the design point in
          ``at_design`` and the state variables in ``at_state``.

        * Multiply the transposed state component of the constraint jacobian
          with the vector stored in ``in_vec``.

        * Store the result in ``out_vec``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        in_vec : BaseVector-like
            Vector to be operated on.
        out_vec : BaseVector-like
            Location where user should store the result.
        """
        pass

    def eval_dFdX(self, at_design, at_state, store_here):
        """
        Evaluate the design component of the objective gradient at the
        design point stored in ``at_design`` and the state variables stored in
        ``at_state``. Store the result in ``store_here``.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        store_here : BaseVector-like
            Location where user should store the result.
        """
        raise NotImplementedError

    def eval_dFdU(self, at_design, at_state, store_here):
        """
        Evaluate the design component of the objective gradient at the
        design point stored in ``at_design`` and the state variables stored in
        ``at_state``. Store the result in ``store_here``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-like
            Current state vector.
        store_here : BaseVector-like
            Location where user should store the result.
        """
        pass

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
        Solve the non linear system at the design point stored in
        ``at_design``. Store the calculated state variables (u) under
        ``result``.

        A simple example is provided below, where the stiffness matrix is
        nonlinearly dependent on the solution.

        .. math:: \mathcal{K}(\mathbf{u})\mathbf{u} = \mathbf{F}

        If the nonlinear solution fails to converge, the user should return a
        ``-1`` integer in order to let Kona decide to backtrack on the
        optimization.

        Similarly, in the case of correct convergence, the user is encouraged
        to return the number of preconditioner calls it took to solve the
        nonlinear system. Kona uses this information to track the
        computational cost of the optimization.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        result : BaseVector-like
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the solution.
        """
        converged = True
        cost = 0
        if converged:
            # You can return the number of preconditioner calls here.
            # This is used by Kona to track computational cost.
            return cost
        else:
            # Must return negative cost to Kona when your system solve fails to
            # converge. This is important because it helps Kona determine when
            # it needs to back-track on the optimization.
            return -1

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        """
        Evaluate the state jacobian, :math:`\\frac{\partial R}{\partial U}`,
        at the design point stored in ``at_design`` and the state variables
        stored in ``at_state``. Solve the linear system
        :math:`\\frac{\partial R}{\partial U}\\mathbf{u}=\\mathbf{v}` where the
        right hand side vector, :math:`\\mathbf{v}`, is the vector stored in
        ``rhs_vec``.

        The solution vector, :math:`\\mathbf{u}`, should be stored at
        ``result``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-line
            Current state vector.
        rhs_vec : BaseVector-like
            Right hand side vector.
        rel_tol : float
            Tolerance that the linear system should be solved to.
        result : BaseVector-like
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the solution.
        """
        return 0

    def solve_adjoint(self, at_design, at_state, rhs_vec, tol, result):
        """
        Evaluate the transposed state jacobian,
        :math:`\\frac{\partial R}{\partial U}^T`, at the design point stored in
        ``at_design`` and the state variables stored in ``at_state``. Solve the
        adjoint system
        :math:`\\frac{\partial R}{\partial U}^T\\mathbf{u}=\\mathbf{v}` where
        the right hand side vector, :math:`\\mathbf{v}`, is the vector stored
        in ``rhs_vec``.

        The solution vector, :math:`\\mathbf{u}`, should be stored at
        ``result``.

        Parameters
        ----------
        at_design : BaseVector-like
            Current design vector.
        at_state : BaseVector-line
            Current state vector.
        rhs_vec : BaseVector-like
            Right hand side vector.
        rel_tol : float
            Tolerance that the linear system should be solved to.
        result : BaseVector-like
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the solution.
        """
        return 0

    def user_info(self, curr_design, curr_state, curr_adj, curr_dual, num_iter):
        """
        Kona will evaluate this method at every outer optimization iteration.
        It can be used to print out useful information to monitor the process,
        or to save design points of the intermediate iterations.

        The current design vector, current state vector and current adjoint
        vector have been made available to the user via the arguments.

        Parameters
        ----------
        curr_design : BaseVector-like
            Current design point.
        curr_state : BaseVector-like
            Current state variables.
        curr_adj : BaseVector-like
            Currently adjoint variables for the objective.
        curr_dual : BaseVector-like
            Current dual vector in storage. (This might be unnecessary!)
        num_iter : int
            Current outer iteration number.
        """
        pass

class UserSolverIDF(UserSolver):
    """
    A modified base class for multidisciplinary problems that adopt the
    IDF (individual discipline feasible) formulation for disciplinary
    coupling.

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
    kona_design : numpy.array
        Kona storage array for vectors of size ``self.num_design``
    kona_state : numpy.array
        Kona storage array for vectors of size ``self.num_state``
    kona_dual : numpy.array
        Kona storage array for vectors of size ``self.num_ceq``

    Parameters
    ----------
    num_real_design : int
        Size of the design space, NOT including target state variables
    num_state : int
        Number of state variables
    num_real_ceq : int (optional)
        Number of equality constraints, NOT including IDF constraints
    """


    def __init__(self, num_design, num_state, num_ceq, allocator=None):
        self.num_real_design = num_design
        self.num_real_ceq = num_ceq
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
