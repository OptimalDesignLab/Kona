from numpy import sqrt

class BaseVector(object):
    """
    An abstract vector class connected to the Kona user memory, containing a
    common set of algebraic member functions.
    
    Attributes
    ----------
    _memory : UserMemory (singleton)
        Pointer to the Kona user memory.
    _index : int
        Storage index for the vector. This is determined by user memory.
        
    Parameters
    ----------
    memory : UserMemory (singleton)
        Pointer to the Kona user memory.
    """
    _flag = None # vector basis flag -- 0 is design, 1 is state, 2 is dual
    
    def __init__(self, memory):
        self._memory = memory
        self._index = self._memory.AssignVector(self)
        
    def __del__(self):
        self._memory.UnassignVector(self)
            
    def __iadd__(self, vec): # this is the += operator
        self._memory.CheckType(vec, type(self))
        self._memory.AXPlusBY(1.0, vec.GetIndex(), 1.0, self._index, self)
        return self
        
    def __isub__(self, vec): # this is the -= operator
        self._memory.CheckType(vec, type(self))
        self._memory.AXPlusBY(-1.0, vec.GetIndex(), 1.0, self._index, self)
        return self
        
    def __imul__(self, val): # this is the *= operator
        if isinstance(val, float):
            self._memory.AXPlusBY(val, self._index, 0.0, -1, self)
            return self
        else:
            raise TypeError('_BaseVector.__imul__() >> ' + \
                            'RHS must be a scalar value.')
                            
    def __idiv__(self, val): # this is the /= operator
        return self.__imul__(1./val)
        
    def Equals(self, rhs): # the = operator cannot be overloaded
        if isinstance(rhs, float):
            self._memory.AXPlusBY(rhs, -1, 0.0, -1, self)
        else:
            self._memory.CheckType(rhs, type(self))
            self._memory.AXPlusBY(0.0, -1, 1.0, rhs.GetIndex(), self)
        
    def EqualsAXPlusBY(self, a, x, b, y): # this performs self = a*x + b*y
        self._memory.CheckType(x, type(self))
        self._memory.CheckType(y, type(self))
        self._memory.AXPlusBY(a, x.GetIndex(), b, y.GetIndex(), self)
        
    def Norm2(self): # this takes the L2 norm of the vector
        val = self._memory.InnerProd(self, self)
        if val < 0:
            raise ValueError('BaseVector.Norm2() >> ' + \
                             'Inner product is negative!')
        else:
            return sqrt(val)
            
    def GetIndex(self): # return the storage index for the vector
        return self._index
        
    def GetFlag(self):
        return self._flag
    
class DesignVector(BaseVector):
    """
    Derived from the base abstracted vector. Contains member functions specific 
    to design vectors.
    """
    _flag = 0
        
    def EqualsBasisVector(self, basis):
        pass
        
    def RestrictToDesign(self):
        self._memory.Restrict(0, self)
    
    def RestrictToTarget(self):
        self._memory.Restrict(1, self)
        
    def Convert(self, dualVec):
        self._memory.ConvertVec(dualVec, self)
        
    def EqualsInitialDesign(self):
        self._memory.SetInitialDesign(self)
        
    def EqualsObjectiveGradient(self, atDesign, atState):
        self._memory.ObjectiveGradient(atDesign, atState, self)
        
    def EqualsReducedGradient(self, atDesign, atState, atAdjoint, workVec):
        pass
        
    def EqualsLagrangianReducedGradient(self, atDesign, atState, atDual, 
                                        atAdjoint, workVec):
        pass
    
class StateVector(BaseVector):
    """
    Derived from the base abstracted vector. Contains member functions specific 
    to state vectors.
    """
    _flag = 1
    
    def EqualsObjectiveGradient(self, atDesign, atState):
        self._memory.ObjectiveGradient(atDesign, atState, self)
    
    def EqualsPDEResidual(self, atDesign, atState):
        self._memory.EvaluatePDEResidual(atDesign, atState, self)
        
    def EqualsPrimalSolution(self, atDesign):
        self._memory.SolvePDE(atDesign, self)
        
class DualVector(BaseVector):
    """
    Derived from the base abstracted vector. Contains member functions specific 
    to state vectors.
    """
    _flag = 2
    
    def Convert(self, designVec):
        self._memory.ConvertVec(self, designVec)
        
    def EqualsConstraints(self, atDesign, atState):
        self._memory.EvaluateConstraints(atDesign, atState, self)