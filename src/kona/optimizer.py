import os.path
from kona.linalg.memory import KonaMemory

class Optimizer(object):
    """
    This is a top-level wrapper for all optimization algorithms contained in
    the Kona library, and also the only class exposed to the outside user.

    Attributes
    ----------
    _memory : KonaMemory
        All-knowing Kona memory controller.
    _algorithm : Algorithm
        Optimization algorithm object.

    Parameters
    ----------
    solver : UserSolver-like
    algorithm : Algorithm
    optns : dict (optional)
    """
    def __init__(self, solver, algorithm, optns=None):
        # modify defaults either from config file or from given dictionary
        self._read_options(optns)

        # initialize optimization memory
        self._memory = KonaMemory(user_solver)

        # set the algorithm
        self._algorithm = algorithm

    def _read_options(self, optns):
        if optns is None:
            if os.path.isfile('kona.cfg'):
                pass
            else:
                self._optns = {}
        else:
            self._optns = optns

    def solve():
        self._algorithm.solve()
