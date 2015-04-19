import sys

from kona.options import BadKonaOption, get_opt

from kona.algorithms.util.merit import ObjectiveMerit

class OptimizationAlgorithm(object):
    """
    Base class for all optimization algorithms.

    Attributes
    ----------
    primal_factory : VectorFactory
        Vector generator for primal space.
    state_factory : VectorFactory
        Vector generator for state space.
    dual_factory : VectorFactory (optional)
        Vector generator for dual space.
    max_iter : int
        Maximum nonlinear iterations for the optimization.
    primal_tol : float
        Relative convergence tolerance for the primal variables.
    constraint_tol : float (optional)
        Relative convergence tolerance for the constraints.
    info_file : file
        File stream for data output.
    merit : MeritFunction-like
        Merit function for the optimization

    Parameters
    ----------
    primal_factory, state_factory, dual_factory : VectorFactory
    optns : dict
    """
    def __init__(self, primal_factory, state_factory, dual_factory=None, optns={}):
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
        # set up the merit function
        merit_optns = get_opt(optns,{},'merit_function')
        merit_type = get_opt(merit_optns, ObjectiveMerit, 'type')
        try:
            self.merit = merit_type(
                primal_factory, state_factory, merit_optns)
        except:
            raise BadKonaOption(optns, 'merit_function', 'type')

    def solve(self):
        """
        Triggers the optimization run.
        """
        raise NotImplementedError # pragma: no cover
