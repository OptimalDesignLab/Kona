import sys

from kona.options import get_opt

class OptimizationAlgorithm(object):
    """
    Base class for all optimization algorithms.

    Parameters
    ----------
    primal_factory, state_factory, eq_factory, ineq_factory : VectorFactory
    optns : dict, optional

    Attributes
    ----------
    primal_factory : :class:`~kona.linalg.memory.VectorFactory`
        Generates primal vectors.
    state_factory : :class:`~kona.linalg.memory.VectorFactory`
        Generates state vectors.
    eq_factory : :class:`~kona.linalg.memory.VectorFactory`
        Generates dual vectors for equality constraints.
    eq_factory : :class:`~kona.linalg.memory.VectorFactory`
        Generates dual vectors for inequality constraints.
    max_iter : int
        Maximum nonlinear iterations for the optimization.
    primal_tol : float
        Relative convergence tolerance for the primal variables.
    constraint_tol : float, optional
        Relative convergence tolerance for the constraints.
    info_file : file
        File stream for data output.
    """
    def __init__(self, primal_factory, state_factory, eq_factory, ineq_factory,
                 optns=None):
        # set up vector factories
        self.primal_factory = primal_factory
        self.state_factory = state_factory
        self.eq_factory = eq_factory
        self.ineq_factory = ineq_factory

        # create empty options dict
        if optns is None:
            self.optns = {}
        else:
            assert type(optns) is dict, "Invalid options! Must be a dictionary."
            self.optns = optns

        # set up max iterations and primal tolerance for convergence
        self.max_iter = get_opt(self.optns, 100, 'max_iter')
        self.primal_tol = get_opt(self.optns, 1e-6, 'opt_tol')
        self.cnstr_tol = get_opt(self.optns, 1e-6, 'feas_tol')

        # set up the info file
        self.info_file = get_opt(self.optns, sys.stdout, 'info_file')
        if isinstance(self.info_file, str):
            self.info_file = self.primal_factory._memory.open_file(
                self.info_file)

        # set up the hist file
        self.hist_file = get_opt(self.optns, 'kona_hist.dat', 'hist_file')
        if isinstance(self.hist_file, str):
            self.hist_file = self.primal_factory._memory.open_file(
                self.hist_file)

    def solve(self):
        """
        Triggers the optimization run.
        """
        raise NotImplementedError # pragma: no cover
