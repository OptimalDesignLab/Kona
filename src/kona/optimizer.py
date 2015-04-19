import os.path

from kona.options import defaults

from kona.user import UserSolver
from kona.algorithms import OptimizationAlgorithm
from kona.linalg.memory import KonaMemory

class Optimizer(object):
    """
    This is a top-level optimization controller. It is intended to be the
    primary means by which a user interacts with Kona.

    Attributes
    ----------
    _memory : KonaMemory
        All-knowing Kona memory controller.
    _algorithm : OptimizationAlgorithm
        Optimization algorithm object.

    Parameters
    ----------
    solver : UserSolver
    algorithm : OptimizationAlgorithm
    optns : dict, optional
    """
    def __init__(self, solver, algorithm, optns=None):
        # complain if solver or algorithm types are wrong
        if not isinstance(solver, UserSolver):
            raise TypeError('Kona.Optimizer() >> ' + \
                            'Unknown solver type!')
        if not isinstance(algorithm, OptimizationAlgorithm):
            raise TypeError('Kona.Optimizer() >> ' + \
                            'Unknown optimization algorithm!')
        # initialize optimization memory
        self._memory = KonaMemory(user_solver)
        # modify defaults either from config file or from given dictionary
        self._read_options(optns)
        # get two mandatory vector factories
        primal_factory = self._memory.primal_factory
        state_factory = self._memory.state_factory
        # check to see if we have constraints
        if solver.num_dual > 0:
            # if we do, recover a dual factory
            dual_factory = self._memory.dual_factory
            # initialize constrained algorithm
            self._algorithm = algorithm(
                primal_factory, state_factory, dual_factory, self.optns)
        else:
            # otherwise initialize unconstrained algorithm
            self._algorithm = algorithm(primal_factory, state_factory, self.optns)
        # finally, when all the initialization is done, allocate memory
        self._memory.allocate_memory()

    def _read_options(self, optns):
        if optns is None:
            if os.path.isfile('kona.cfg'):
                raise NotImplementedError
            else:
                self._optns = defaults
        else:
            self._optns = optns

    def solve():
        self._algorithm.solve()
