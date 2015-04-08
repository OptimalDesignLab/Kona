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
        ``self.kona_design[at_design]`` and the state variables stored at
        ``self.kona_state[at_state]``.

        If ``at_state == -1``, the state variables should be calculated at the
        provided design point before being used in the objective function. In
        this case, the user should keep track of

        This function should return a tuple containing the objective function
        value as the first element, and the number of preconditioner calls as
        the second.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.

        Returns
        -------
        tuple : Result of the operation. Contains the objective value as the
        first element, and the number of preconditioner calls used as the
        second.
        """
        raise NotImplementedError

    def eval_residual(self, at_design, at_state, result):
        """
        Evaluate the linearized system (PDE) using the design point stored in
        ``self.kona_design[x]`` and the state variables stored in
        ``self.kona_state[y]``. Put the residual vector in
        ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        result : int
            Location where user should store the result.
        """
        pass

    def eval_ceq(self, at_design, at_state, result):
        """
        Evaluate the vector of equality constraints using the design
        variables stored at `self.kona_design[x]`` and the state
        variables stored at ``self.kona_state[y]``. Store the constraint
        residual at ``self.kona_dual[result].

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        result : int
            Location where user should store the result.
        """
        pass

    def multiply_dRdX(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Update the linearized system jacobian using the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the design component of the system jacobian with the vector
          stored in ``self.kona_design[vec]``.

        * Store the result in ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        pass

    def multiply_dRdU(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Linearize the system about the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the state component of the system jacobian with the vector
          stored in ``self.kona_state[vec]``.

        * Store the result in ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        pass

    def multiply_dRdX_T(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Linearize the system about the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the transposed design component of the system jacobian with
          the vector stored in ``self.kona_state[vec]``.

        * Store the result in ``self.kona_design[result]``.

        .. note::

            Must always store a result even when it isn't implemented. Use a
            zero vector of length ``self.num_design`` for this purpose.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        self.kona_design[result] = np.zeros(self.num_design)

    def multiply_dRdU_T(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Linearize the system about the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the transposed state component of the system jacobian with
          the vector stored in ``self.kona_state[vec]``.

        * Store the result in ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        pass

    def build_precond(self):
        """
        OPTIONAL: Build the system preconditioner.
        """
        pass

    def multiply_precond(self, at_design, at_state, vec, result):
        """
        OPTIONAL: Apply the preconditioner to the vector at
        ``self.kona_state[vec]`` and store the result in
        ``self.kona_state[result]``. If the preconditioner has to be
        linearized, use the design and state vectors provided in
        ``self.kona_design[at_design]`` and ``self.kona_state[at_state]``.

        Parameters
        ----------
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the operation.
        """
        return 0

    def multiply_precond_T(self, at_design, at_state, vec, result):
        """
        OPTIONAL: Apply the transpose of the preconditioner to the vector at
        ``self.kona_state[vec]`` and store the result in
        ``self.kona_state[result]``. If the preconditioner has to be
        linearized, use the design and state vectors provided in
        ``self.kona_design[at_design]`` and ``self.kona_state[at_state]``.

        Parameters
        ----------
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the operation.
        """
        return 0

    def multiply_dCdX(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Linearize the equality constraints about the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the design component of the constraint jacobian with
          the vector stored in ``self.kona_design[vec]``.

        * Store the result in ``self.kona_dual[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        pass

    def multiply_dCdU(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Linearize the equality constraints about the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the state component of the constraint jacobian with
          the vector stored in ``self.kona_state[vec]``.

        * Store the result in ``self.kona_dual[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        pass

    def multiply_dCdX_T(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Linearize the equality constraints about the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the transposed design component of the constraint jacobian
          with the vector stored in ``self.kona_dual[vec]``.

        * Store the result in ``self.kona_design[result]``.

        .. note::

            Must always store a result even when it isn't implemented. Use a
            zero vector of length ``self.num_design`` for this purpose.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        self.kona_design[result] = np.zeros(self.num_design)

    def multiply_dCdU_T(self, at_design, at_state, vec, result):
        """
        Perform the following tasks:

        * Linearize the equality constraints about the design point in
          ``self.kona_design[at_design]`` and the state variables in
          ``self.kona_state[at_state]``.

        * Multiply the transposed state component of the constraint jacobian
          with the vector stored in ``self.kona_dual[vec]``.

        * Store the result in ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        vec : int
            Storage index for the vector to be operated on.
        result : int
            Location where user should store the result.
        """
        pass

    def eval_dFdX(self, at_design, at_state, result):
        """
        Evaluate the design component of the objective gradient at the
        design point stored in ``self.kona_design[vec]`` and store the
        result in ``self.kona_design[result]``.

        .. note::

            This method must be implemented for any problem type.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        result : int
            Location where user should store the result.
        """
        raise NotImplementedError

    def eval_dFdU(self, at_design, at_state, result):
        """
        Evaluate the state component of the objective gradient at the
        design point stored in ``self.kona_state[vec]`` and store the
        result in ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        result : int
            Location where user should store the result.
        """
        pass

    def init_design(self, store):
        """
        Initialize the first design point. Store the design vector at
        ``self.kona_design[store]``. The optimization will start from
        here.

        .. note::

            This method must be implemented for any problem type.
        """
        raise NotImplementedError

    def solve_system(self, at_design, result):
        """
        Solve the non linear system at the design point stored in
        ``self.kona_design[at_design]. Store the calculated state variables (u)
        under ``self.kona_state[result]``.

        .. math:: \mathcal{K}\mathbf{u} = \mathbf{F}

        If the linearized system solution requires an iterative method, the
        user should return a ``-1`` integer when this solution fails to
        converge. Kona can use this information to backtrack on the
        optimization and ensure that the non-linear system is consistent at
        each step.

        Similarly, in the case of correct convergence, the user is encouraged
        to return the number of preconditioner calls it took to solve the
        linearized system. Kona uses this information to track the
        computational cost of the optimization.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        result : int
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the solution.
        """
        converged = True
        if converged:
            # You can return the number of preconditioner calls here.
            # This is used by Kona to track computational cost.
            return 0
        else:
            # Must return negative cost to Kona when your system solve fails to
            # converge. This is important because it helps Kona determine when
            # it needs to back-track on the optimization.
            return -1

    def solve_linearsys(self, at_design, at_state, rhs, tol, result):
        """
        Evaluate the state jacobian, ``A``, at the design point stored in
        ``self.kona_design[at_design]`` and the state variables stored in
        ``self.kona_state[at_state]``. Solve the linear system
        :math:`A\\mathbf{u}=\\mathbf{v}` where the right hand side vector,
        :math:`\\mathbf{v}`, is the vector stored in ``self.kona_state[rhs]``.

        The solution vector, :math:`\\mathbf{u}`, should be stored at
        ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        rhs : int
            Storage index for the right hand side vector.
        tol : float
            Tolerance that the linear system should be solved to.
        result : int
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the solution.
        """
        return 0

    def solve_adjoint(self, at_design, at_state, rhs, tol, result):
        """
        Evaluate the state jacobian, ``A``, at the design point stored in
        ``self.kona_design[at_design]`` and the state variables stored in
        ``self.kona_state[at_state]``. Solve the adjoint system
        :math:`A^T\\mathbf{u}=\\mathbf{v}`
        where the right hand side vector, :math:`\\mathbf{v}`, is:

            1. If ``rhs == -1``, then :math:`\\mathbf{v}` is the negative
            derivative of the objective function with respect to the state
            variables (:math:`\\mathbf{v} = -\\frac{dJ}{du}).

            2. Otherwise, :math:`\\mathbf{v}` is the vector that is stored in
            ``self.kona_state[rhs]``.

        The solution vector, :math:`\\mathbf{u}`, should be stored at
        ``self.kona_state[result]``.

        Parameters
        ----------
        at_design : int
            Storage index for the current design vector.
        at_state : int
            Storage index for the current state vector.
        rhs : int
            Storage index for the right hand side vector.
        tol : float
            Tolerance that the adjoint system should be solved to.
        result : int
            Location where user should store the result.

        Returns
        -------
        int : Number of preconditioner calls required for the solution.
        """
        return 0

    def user_info(self, curr_design, curr_state, curr_adj, curr_dual, num_iter):
        """
        Kona will evaluate this method at every optimization iteration. It
        can be used to print out useful information to monitor the process, or
        to save design points of the intermediate iterations.

        The storage indexes for current design vector, current state vector and
        current adjoint vector have been made available to the user. These can
        be accessed via:

        Current Design Variables = ``self.kona_design[curr_design]``
        Current State Variables = ``self.kona_state[curr_state]``
        Current Adjoint Variables ``self.kona_state[curr_adj]``
        Optimization iteration number = ``num_iter``
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
        super(BaseAllocatorIDF, self).__init__(
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
            vec[self.num_real_design:] = 0.
        elif opType == 1:
            vec[:self.num_real_design] = 0.
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
