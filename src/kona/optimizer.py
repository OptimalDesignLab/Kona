from kona.linalg.memory import KonaMemory

class Optimizer(object):
    """
    This is a top-level wrapper for all optimization algorithms contained in
    the Kona library, and also the only class exposed to the outside user.
    """

    def __init__(self, userObj, optns=None):
        # initialize default options
        self.options = {}

        # modify defaults either from config file or from given dictionary
        self._readOptions(optns)

        # calculate memory requirements
        num_primal_vec, num_state_vec, num_dual_vec = _memoryRequirements(self)

        # initialize optimization memory
        self.memory = KonaMemory(userObj, num_primal_vec, num_state_vec, num_dual_vec)

    def _readOptions(self, optns):
        pass

    def _memoryRequirements(self):
        return (0, 0, 0)

    def Optimize():
        pass
