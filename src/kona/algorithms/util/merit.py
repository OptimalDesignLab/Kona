import sys

from kona.linalg import objective_value
from kona.linalg.solvers.util import EPS
from kona.linalg.matrices.common import ActiveSetMatrix, dRdX

class MeritFunction(object):
    """
    Base class for all merit functions.

    Attributes
    ----------
    primal_factory : VectorFactory
        Generator for new primal vectors.
    state_factory : VectorFactory
        Generator for new state vectors.
    out_file : file
        File stream for writing data.
    _allocated : boolean
        Flag to track whether merit function memory has been allocated.

    Parameters
    ----------
    primal_factory : VectorFactory
    state_factory : VectorFactory
    optns : dict
    out_file : file
    """
    def __init__(self, primal_factory, state_factory,
                 optns={}, out_file=sys.stdout):
        self.primal_factory = primal_factory
        self.state_factory = state_factory
        self.out_file = out_file
        self._allocated = False

    def reset(self, search_dir, x_start, u_start, p_dot_grad=None):
        """
        Reset the merit function at a new design and state point.

        If merit memory is not yet allocated, this function should also do that.

        Parameters
        ----------
        search_dir : PrimalVector
            Search direction vector in the primal space.
        x_start : PrimalVector
            Initial primal vector.
        u_start : StateVector
            State vector corresponding to ``x_start``.
        p_dot_grad : float (optional)
            Value of :math:`\\langle p, \\nabla f \\rangle` at ``x_start``.
        """
        raise NotImplementedError # pragma: no cover

    def eval_func(self, alpha):
        """
        Evaluate merit function value at ``alpha``

        Parameters
        ----------
        alpha : float

        Returns
        -------
        float : Value of merit function ``alpha``.
        """
        raise NotImplementedError # pragma: no cover

    def eval_grad(self, alpha):
        """
        Evaluate merit function gradient :math:`\\langle p, \\nabla f \\rangle`
        at the given ``alpha``

        .. note::

            This method can either ``pass`` or ``return 0`` for gradient-free
            merit functions.

        Parameters
        ----------
        alpha : float

        Returns
        -------
        float : Value of :math:`\\langle p, \\nabla f \\rangle` at ``alpha``.
        """
        return 0 # pragma: no cover

class ObjectiveMerit(MeritFunction):
    """
    Merit function for line searches applied to the raw objective.

    Other, more complicated merit functions can be derived from this.

    Attributes
    ----------
    last_func_alpha : float
        Last value of alpha for which objective value is evaluated.
    last_grad_alpha : float
        Last value of alpha for which objective grad is evaluated.
    func_val : float
        Value of the objective at ``last_func_alpha``.
    p_dot_grad : float
        Value of :math:`\\langle p, \\nabla f \\rangle` at ``last_grad_alpha``.
    x_start : PrimalVector
        Initial position of the primal variables, where :math:`\\alpha = 0`.
    x_trial : PrimalVector
        Trial position of the primal variables at a new alpha.
    u_trial : StateVector
        Trial position of the state variables at a new alpha.
    search_dir : PrimalVector
        The search direction vector.
    state_work : StateVector
        Work vector for state operations.
    adjoint_work : StateVector
        Work vector for adjoint operations.
    design_work : PrimalVector
        Work vector for primal operations.
    """

    def __init__(self, primal_factory, state_factory,
                 optns={}, out_file=sys.stdout):
        super(ObjectiveMerit, self).__init__(primal_factory, state_factory,
                                             optns, out_file)
        self.primal_factory.request_num_vectors(2)
        self.state_factory.request_num_vectors(2)

    def reset(self, search_dir, x_start, u_start, p_dot_grad):
        # if the internal vectors are not allocated, do it now
        if not self._allocated:
            self.x_trial = self.primal_factory.generate()
            self.primal_work = self.primal_factory.generate()
            self.state_work = self.state_factory.generate()
            self.adjoint_work = self.state_factory.generate()
            self._allocated = True
        # store information for the new point the merit function is reset at
        self.search_dir = search_dir
        self.x_start = x_start
        self.u_trial = u_start
        self.func_val = objective_value(x_start, u_start)
        self.p_dot_grad = p_dot_grad
        self.last_func_alpha = 0.0
        self.last_grad_alpha = 0.0

    def eval_func(self, alpha):
        # do calculations only if alpha changed significantly
        if abs(alpha - self.last_func_alpha) > EPS:
            # calculate the trial primal and state vectors
            self.x_trial.equals_ax_p_by(
                1.0, self.x_start, alpha, self.search_dir)
            self.u_trial.equals_primal_solution(self.x_trial)
            # calculate and return the raw objective function
            self.func_val = objective_value(self.x_trial, self.u_trial)
            # store last used alpha
            self.last_func_alpha = alpha

        return self.func_val

    def eval_grad(self, alpha):
        # do calculations only if alpha changed significantly
        if abs(alpha - self.last_grad_alpha) > EPS:
            # calculate the trial primal and state vectors
            self.x_trial.equals_ax_p_by(
                1.0, self.x_start, alpha, self.search_dir)
            self.u_trial.equals_primal_solution(self.x_trial)
            # calculate objective partial
            self.primal_work.equals_objective_partial(
                self.x_trial, self.u_trial)
            # add contribution from objective partial
            self.p_dot_grad = self.search_dir.inner(self.primal_work)
            # calculate adjoint
            self.adjoint_work.equals_adjoint_solution(
                self.x_trial, self.u_trial, self.state_work)
            # create dR/dX jacobian wrapper
            jacobian = dRdX(self.x_trial, self.u_trial)
            # multiply the adjoint by dR/dX^T and store into primal work
            jacobian.T.product(self.adjoint_work, self.primal_work)
            # add contribution from non-linear state changes
            self.p_dot_grad += self.search_dir.inner(self.primal_work)
            # store last used alpha
            self.last_grad_alpha = alpha

        return self.p_dot_grad

class AugmentedLagrangian(MeritFunction):

    def __init__(self, primal_factory, state_factory, dual_factory,
                 optns={}, out_file=sys.stdout):
        super(AugmentedLagrangian, self).__init__(
            primal_factory, state_factory, optns, out_file)
        self.dual_factory = dual_factory
        self.primal_factory.request_num_vectors(1)
        self.state_factory.request_num_vectors(1)
        self.dual_factory.request_num_vectors(2)

    def reset(self, kkt_start, u_start, search_dir, p_dot_grad, mu):
        # if the internal vectors are not allocated, do it now
        if not self._allocated:
            self.x_trial = self.primal_factory.generate()
            self.u_trial = self.state_factory.generate()
            self.dual_frozen = self.dual_factory.generate()
            self.dual_work = self.dual_factory.generate()
            self._allocated = True
        # store information for the new point the merit function is reset at
        self.search_dir = search_dir
        self.x_trial.equals(kkt_start._primal)
        self.u_trial.equals(u_start)
        self.dual_frozen.equals(kkt_start._dual)
        self.p_dot_grad = p_dot_grad
        self.mu = mu
        # evaluate the function value
        self.dual_work.equals_constraints(self.x_trial, self.u_trial)
        ActiveSetMatrix(self.dual_work).product(
            self.dual_work, self.dual_work)
        self.func_val = \
            objective_value(self.x_trial, self.u_trial) \
            + self.dual_frozen.inner(self.dual_work) \
            + 0.5*self.mu*(self.dual_work.norm2**2)
        self.last_func_alpha = 0.0

    def eval_func(self, alpha):
        if abs(alpha - self.last_func_alpha) > EPS:
            self.x_trial.equals_ax_p_by(
                1., self.x_trial, alpha, self.search_dir._primal)
            self.u_trial.equals_primal_solution(self.x_trial)
            self.dual_work.equals_constraints(self.x_trial, self.u_trial)
            ActiveSetMatrix(self.dual_work).product(
                self.dual_work, self.dual_work)
            self.func_val = \
                objective_value(self.x_trial, self.u_trial) \
                + self.dual_frozen.inner(self.dual_work) \
                + 0.5*self.mu*(self.dual_work.norm2**2)
            self.last_func_alpha = alpha

        return self.func_val
