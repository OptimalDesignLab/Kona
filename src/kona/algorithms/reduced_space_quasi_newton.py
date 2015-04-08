import sys

from kona.linalg.matrices.composite import LimitedMemoryBFGS, LimitedMemorySR1
from kona.algorithms.util.linesearch import StrongWolfe, BackTracking, ObjectiveMerit
from kona import BadKonaOption

class ReducedSpaceQuasiNewton(object):
    """
    Unconstrained optimization using quasi-Newton in the reduced space.
    """

    def __init__(self, primal_factory, dual_factory, state_factory, optns,
                 out_file=sys.stdout):
        # vectors required in solve() method
        primal_factory.request_num_vectors(6)
        state_factory.request_num_vectors(2)
        # set the type of quasi-Newton method
        if optns['quasi_newton']['type'] == 'lbfgs':
            self.quasi_newton = LimitedMemoryBFGS(primal_factory, optns['quasi_newton'],
                                                  out_file)
        elif optns['quasi_newton']['type'] == 'sr1':
            self.quasi_newton = LimitedMemorySR1(primal_factory, optns['quasi_newton'],
                                                 out_file)
        else:
            raise BadKonaOption(optns, ('quasi_newton', 'type'))
        # set the type of line-search algorithm
        if optns['line_search']['type'] == 'wolfe':
            self.line_search = StrongWolfe(optns['line_search'], out_file)
        elif optns['line_search']['type'] == 'back_track':
            self.line_search = BackTracking(optns['line_search'], out_file)
        else:
            raise BadKonaOption(optns, ('line_search', 'type'))
        # define the merit function (which is always the objective itself here)
        self.merit = ObjectiveMerit(optns['merit'], vector_factory, outfile)

    def solve():
        pass
