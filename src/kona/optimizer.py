
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
    def __init__(self, solver, algorithm, optns={}):

        # initialize optimization memory
        self._memory = KonaMemory(solver)

        # set default file handles
        self._optns = {
            'info_file' : 'kona_info.dat',
            'hist_file' : 'kona_hist.dat',
            'krylov' : {
                'out_file' : 'kona_krylov.dat',
            },
            'verify' : {
                'out_file' : 'kona_verify.dat',
            }
        }

        # process the final options
        if not isinstance(optns, dict):
            raise TypeError('Kona.Optimizer >> Options must be a dictionary!')
        self._process_options(optns)

        # get vector factories
        primal_factory = self._memory.primal_factory
        state_factory = self._memory.state_factory
        eq_factory = self._memory.eq_factory
        ineq_factory =  self._memory.ineq_factory

        # initialize the optimization algorithm
        self._algorithm = algorithm(
            primal_factory, state_factory, eq_factory, ineq_factory,
            self._optns)

    def _process_options(self, optns):
        # this is a recursive dictionary merge function
        def update(d, u):
            for k, v in u.iteritems():
                if isinstance(v, collections.Mapping):
                    r = update(d.get(k, {}), v)
                    d[k] = r
                else:
                    d[k] = u[k]
            return d
        # merge user dictionary with default file names
        self._optns = update(self._optns, optns)
        # open the files on the master (zero) rank
        self._optns['info_file'] = \
            self._memory.open_file(self._optns['info_file'])
        self._optns['hist_file'] = \
            self._memory.open_file(self._optns['hist_file'])
        self._optns['krylov']['out_file'] = \
            self._memory.open_file(self._optns['krylov']['out_file'])
        self._optns['verify']['out_file'] = \
            self._memory.open_file(self._optns['verify']['out_file'])

    def set_design_bounds(self, lower=None, upper=None):
        """
        Define lower and upper design bounds.

        Parameters
        ----------
        lower : int
            Lower bound for design variables.
        upper : int
            Upper bound for design variables.
        """
        # make sure user provided some value
        if lower is not None and upper is None:
            raise ValueError("Must specify at least one bound!")
        # assign the bounds
        if lower is not None:
            assert isinstance(lower, (np.float, np.int))
            self._memory.design_lb = lower
        if upper is not None:
            assert isinstance(upper, (np.float, np.int))
            self._memory.design_ub = upper

    def solve(self):
        self._memory.allocate_memory()
        self._algorithm.solve()

# package imports at the bottom to prevent circular import errors
import collections
import numpy as np
from kona.user import UserSolver
from kona.linalg.memory import KonaMemory