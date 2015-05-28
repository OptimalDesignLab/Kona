import os.path

from kona.options import defaults

from kona.user import UserSolver
from kona.algorithms import OptimizationAlgorithm
from kona.linalg.memory import KonaMemory

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
        # get two mandatory vector factories
        primal_factory = self._memory.primal_factory
        state_factory = self._memory.state_factory
        # check to see if we have constraints
        if solver.num_dual > 0:
            # if we do, recover a dual factory
            dual_factory = self._memory.dual_factory
            # initialize constrained algorithm
            self._algorithm = algorithm(
                primal_factory, state_factory, dual_factory, self._optns)
        else:
            # otherwise initialize unconstrained algorithm
            self._algorithm = algorithm(primal_factory, state_factory, self._optns)

    def _read_options(self, optns):
        self._optns = defaults.copy()
        if isinstance(optns, dict):
            self._optns.update(optns)
        else:
            if os.path.isfile('kona.cfg'):
                raise NotImplementedError

    def solve(self):
        self._memory.allocate_memory()
        self._algorithm.solve()

    # def verify(self):
    #
    #     # get references to vector factories
    #     pf = self._memory.primal_factory
    #     sf = self._memory.state_factory
    #     constrained = False
    #     if self._memory.solver.num_dual > 0:
    #         constrained = True
    #         df = self._memory.dual_factory
    #
    #     # request vectors
    #     pf.request_num_vectors(5)
    #     sf.request_num_vectors(5)
    #     if constrained:
    #         df.request_num_vectors(5)
    #
    #     # allocate memory
    #     self._memory.allocate_memory()
    #
    #     # generate vectors
    #     X = pf.generate()
    #     primal_work = pf.generate()
    #     state = sf.generate()
    #     adjoint = sf.generate()
    #     state_work = sf.generate()
    #
    #     # start testing objective partials
    #     print 'Objective Partials:'
    #     print '=============================================================='
    #
    #     # calculate [1]^T*dF/dX*[1] partial
    #     epsilon_fd = 1e-5
    #     X.equals_init_design()
    #     state.equals_primal_solution(X)
    #     obj_val = objective_value(X, state)
    #     primal_work.equals(1.0)
    #     X.equals_ax_p_by(1.0, x, epsilon_fd, primal_work)
    #     state.equals_primal_solution(X)
    #     obj_val_pert = objective_value(X, state)
    #     dFdX_pert = (obj_val_pert - obj)/epsilon_fd
