from common import Singleton
from usermemory import UserMemory

class Optimizer(Singleton):
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
        numDesignVec, numStateVec, numDualVec = self_memoryRequirements(self)
        
        # initialize optimization memory
        self.memory = UserMemory(userObj, numDesignVec, numStateVec, numDualVec)
        
    def _readOptions(self, optns):
        pass
        
    def _memoryRequirements(self):
        return (0, 0, 0)
        
    def Optimize():
        pass