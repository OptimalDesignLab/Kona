"""This module contains the core template class from which Kona users can
derive their own problem definitions. Instructions on which base class to
derive from are provided in the class docstrings below.

When deriving user-defined problem objects, the default method names and
arguments defined below should be preserved exactly as shown. The user should
only change the contents of these methods to perform the corresponding tasks
required by Kona for the optimization. To that end, the user is free to add new
intermediate methods. Kona will only utilize methods that are part of this base
class.

.. note::

   All of the matrix-vector multiplication methods can be implemented
   matrix-free. The choice is left up to the user.

"""

import numpy as np

class UserTemplate(object):
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
    kona_design : numpy.array
        Kona storage array for vectors of size ``self.num_design``
    kona_state : numpy.array
        Kona storage array for vectors of size ``self.num_state``
    kona_dual : numpy.array
        Kona storage array for vectors of size ``self.num_ceq``
        
    Parameters
    ----------
    num_design : int (optional)
        Number of design variables.
    num_state : int (optional)
        Number of state variables.
    num_ceq : int (optional)
        Number of equality constraints.
    """

    def __init__(self, num_design=0, num_state=0, num_ceq=0):
        self.num_design = num_design
        self.num_state = num_state
        self.num_ceq = num_ceq
        
    def get_rank(self):
        """
        Return the processor rank.
        
        For serial codes, this should always return 0 (zero). This rank is used 
        to ensure that print statements originate only from the root (0 rank) 
        processor.
        
        Returns
        -------
        int : Processor rank.
        """
        return 0

    def alloc_memory(self, num_design_vec, num_state_vec, num_dual_vec):
        """
        Create the storage arrays required by Kona.

        Parameters
        ----------
        num_design_vec : int
            Number of storage vectors for design variables, each with a length
            of ``self.num_design``.
        num_state_vec : int
            Number of storage vectors for state variables, each with a length
            of ``self.num_state``.
        num_dual_vec : int
            Number of storage vectors for ceq (constraint) variables, each
            with a length of ``self.num_state``.
            
        Returns
        -------
        int : Error code for the operation. 0 (zero) means no error.
        """
        self.kona_design = np.zeros((num_design_vec, self.num_design))
        self.kona_state = np.zeros((num_state_vec, self.num_state))
        self.kona_dual = np.zeros((num_dual_vec, self.num_ceq))
        return 0

    def axpby_d(self, a, vec1, b, vec2, result):
        """
        User-defined linear algebra method for scalar multiplication and
        vector addition.

        .. math:: a\mathbf{x} + b\mathbf{y}

        Parameters
        ----------
        a, b : double
            Multiplication coefficients.
        vec1, vec2 : int
            Storage indexes for the vectors to be operated on.
        result : int
            Storage index for the result of the operation.
        """
        #print "axpby_d : %.3f * vec %i + %.2f * vec %i" % (a, vec1, b, vec2)
        if vec1 == -1:
            if vec2 == -1: # if indexes for both vectors are -1
                # answer is a vector of ones scaled by a
                out = a*np.ones(self.num_design)
            else: # if only the index for vec1 is -1
                # answer is vec2 scaled by b
                if b == 0.:
                    out = np.zeros(self.num_design)
                else:
                    y = self.kona_design[vec2]
                    out = b*y
        elif vec2 == -1: # if only the index for vec2 is -1
            # answer is vec1 scaled by a
            if a == 0.:
                out = np.zeros(self.num_design)
            else:
                x = self.kona_design[vec1]
                out = a*x
        else:
            # otherwise perform the full a*vec1 + b*vec2 operation
            x = self.kona_design[vec1]
            y = self.kona_design[vec2]
            if a == 0.:
                if b == 0.:
                    out = self.zeros(self.num_design)
                else:
                    out = b*y
            else:
                if b == 0.:
                    out = a*x
                else:
                    out = a*x + b*y
        # write the result into the designated location
        self.kona_design[result] = out

    def axpby_s(self, a, vec1, b, vec2, result):
        """
        See ``axpby_d``. Perform the same tasks for vectors of size
        ``self.num_state``.
        """
        pass

    def axpby_ceq(self, a, vec1, b, vec2, result):
        """
        See ``axpby_d``. Perform the same tasks for vectors of size
        ``self.num_ceq``.
        """
        pass

    def inner_prod_d(self, vec1, vec2):
        """
        User-defined linear algebra method for a vector inner product.

        Parameters
        ----------
        vec1, vec2 : int
            Storage indexes for the vectors to be operated on.

        Returns
        -------
        float : Result of the operation.
        """
        return np.inner(self.kona_design[vec1], self.kona_design[vec2])

    def inner_prod_s(self, vec1, vec2):
        """
        See ``inner_prod_d``. Must return ``0.0`` when not implemented.
        """
        return 0.0

    def inner_prod_ceq(self, vec1, vec2):
        """
        See ``inner_prod_d``. Must return ``0.0`` when not implemented.
        """
        return 0.0

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
        ``self.kona_design[vec1]`` and the state variables stored in
        ``self.kona_state[vec2]``. Put the residual vector in
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
        variables stored at `self.kona_design[vec1]`` and the state
        variables stored at ``self.kona_state[vec2]``. Store the constraint
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

    def multiply_jac_d(self, at_design, at_state, vec, result):
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

    def multiply_jac_s(self, at_design, at_state, vec, result):
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

    def multiply_tjac_d(self, at_design, at_state, vec, result):
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

    def multiply_tjac_s(self, at_design, at_state, vec, result):
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

    def multiply_tprecond(self, at_design, at_state, vec, result):
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

    def multiply_ceqjac_d(self, at_design, at_state, vec, result):
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

    def multiply_ceqjac_s(self, at_design, at_state, vec, result):
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

    def multiply_tceqjac_d(self, at_design, at_state, vec, result):
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

    def multiply_tceqjac_s(self, at_design, at_state, vec, result):
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

    def eval_grad_d(self, at_design, at_state, result):
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

    def eval_grad_s(self, at_design, at_state, result):
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
        Solve the linearized system at the design point stored in
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

class UserTemplateIDF(UserTemplate):
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

    def __init__(self, num_real_design, num_state, num_real_ceq=0):
        # variables with the _real_ designation do not include information
        # about the IDF target state variables
        self.num_real_design = num_real_design
        self.num_real_ceq = num_real_ceq
        num_design = num_real_design + num_state
        num_ceq = num_real_ceq + num_state
        UserTemplate.__init__(self, num_design, num_state, num_ceq)

    def restrict_design(self, type, target):
        """
        If operation type is 0 (``type == 0``), set the target state variables 
        to zero.
        
        If operation type is 1 (``type == 1``), set the real design variables 
        to zero.
        
        This operation should be performed on the design vector stored at 
        ``self.kona_design[target]``.
        
        Parameters
        ----------
        type : int
            Operation type flag.
        target : int
            Index of the design vector to be operated on.
        """
        if type == 0:
            self.kona_design[target, self.num_real_design:] = 0.
        elif type == 1:
            self.kona_design[target, :self.num_real_design] = 0.
        else:
            raise ValueError('Unexpected type in restrict_design()!')

    def copy_dual_to_targstate(self, take_from, copy_to):
        """
        Take the target state variables from dual storage, 
        ``self.kona_dual[take_from]``, and put them into design storage, 
        ``self.kona_design[copy_to]``. Also set the real design variables to 
        zero.
        
        Parameters
        ----------
        take_from : int
            Dual index from where target state variables should be taken.
        copy_to : int
            Design index to which target state variables should be copied.
        """
        self.kona_design[copy_to, :self.num_real_design] = 0.
        self.kona_design[copy_to, self.num_real_design:] = \
            self.kona_dual[take_from, self.num_real_ceq:]

    def copy_targstate_to_dual(self, take_from, copy_to):
        """
        Take the target state variables from design storage, 
        ``self.kona_design[take_from]``, and put them into dual storage, 
        ``self.kona_dual[copy_to]``.
        
        Parameters
        ----------
        take_from : int
            Design index from where target state variables should be taken.
        copy_to : int
            Dual index to which target state variables should be copied.
        """
        self.kona_dual[copy_to, self.num_real_ceq:] = \
            self.kona_design[take_from, self.num_real_design:]
