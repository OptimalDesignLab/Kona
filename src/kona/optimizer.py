import os.path

from kona.user import UserSolver
from kona.algorithms import OptimizationAlgorithm, Verifier
from kona.linalg.memory import KonaMemory
from kona.options import defaults

#from kona.linalg.vectors import objective_value

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
        # initialize optimization memory
        self._memory = KonaMemory(solver)
        # modify defaults either from config file or from given dictionary
        self._read_options(optns)
        # get vector factories
        primal_factory = self._memory.primal_factory
        state_factory = self._memory.state_factory
        if solver.num_dual > 0:
            dual_factory = self._memory.dual_factory
        else:
            dual_factory = None
        # check if this is a verification
        if algorithm is Verifier:
            self._optns['verify']['out_file'] = \
                self._memory.open_file(self._optns['verify']['out_file'])
            self._algorithm = Verifier(
                [primal_factory, state_factory, dual_factory],
                solver, self._optns['verify'])
        else:
            # otherwise initialize the optimization algorithm
            if dual_factory is not None:
                self._algorithm = algorithm(
                    primal_factory, state_factory, dual_factory, self._optns)
            else:
                self._algorithm = algorithm(
                    primal_factory, state_factory, self._optns)

    def _read_options(self, optns):
        # get default options
        self._optns = defaults.copy()
        # update default options with the given dictionary
        if isinstance(optns, dict):
            self._optns.update(optns)
            self._optns['info_file'] = \
                self._memory.open_file(self._optns['info_file'])
            self._optns['hist_file'] = \
                self._memory.open_file(self._optns['hist_file'])
            self._optns['krylov']['out_file'] = \
                self._memory.open_file(self._optns['krylov']['out_file'])
        else:
            if os.path.isfile('kona.cfg'):
                raise NotImplementedError

    def solve(self):
        self._memory.allocate_memory()
        self._algorithm.solve()
