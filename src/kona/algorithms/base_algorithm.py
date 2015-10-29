import sys

from kona.options import get_opt

class OptimizationAlgorithm(object):
    """
    Base class for all optimization algorithms.

    Parameters
    ----------
    primal_factory : VectorFactory
        PrimalVector factory.
    state_factory : VectorFactory
        StateVector factory.
    dual_factory : VectorFactory
        DualVector factory.
    optns : dict
        Options dictionary.

    Attributes
    ----------
    primal_factory : VectorFactory
        Generates primal vectors.
    state_factory : VectorFactory
        Generates state vectors.
    dual_factory : VectorFactory
        Generates dual vectors.
    max_iter : int
        Maximum nonlinear iterations for the optimization.
    primal_tol : float
        Relative convergence tolerance for the primal variables.
    constraint_tol : float, optional
        Relative convergence tolerance for the constraints.
    info_file : file
        File stream for data output.
    merit_func : MeritFunction
        Merit function for the optimization
    """
    def __init__(self, primal_factory, state_factory, dual_factory,
                 optns={}):
        # set up vector factories
        self.primal_factory = primal_factory
        self.state_factory = state_factory
        if dual_factory is not None:
            self.dual_factory = dual_factory
            self.constraint_tol = get_opt(optns, 1e-8, 'constraint_tol')

        # set up max iterations and primal tolerance for convergence
        self.max_iter = get_opt(optns, 100, 'max_iter')
        self.primal_tol = get_opt(optns, 1e-8,'primal_tol')

        # set up the info file
        self.info_file = get_opt(optns, sys.stdout, 'info_file')
        if isinstance(self.info_file, str):
            self.info_file = open(self.info_file,'w')

        # set up the hist file
        self.hist_file = get_opt(optns, 'kona_hist.dat', 'hist_file')
        if isinstance(self.hist_file, str):
            self.hist_file = open(self.hist_file, 'w')

    def solve(self):
        """
        Triggers the optimization run.
        """
        raise NotImplementedError # pragma: no cover
