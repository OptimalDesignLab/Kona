from numpy import sqrt
from kona.linalg.vectors.common import DesignVector, DualVector

class ReducedKKTVector(object):
    
    def __init__(self, memory, designVec, dualVec):
        self._memory = memory
        if isinstance(designVec, DesignVector):
            self._design = designVec
        else:
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Unidentified design vector.')
        if isinstance(dualVec, DualVector):
            self._dual = dualVec
        else:
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Unidentified dual vector.')
                            
    def _checkType(self, vec):
        if not isinstance(vec, type(self)):
            raise TypeError('ReducedKKTVector.__iadd__() >> ' + \
                            'Wrong vector type. Must be %s' % type(self))
                            
    def __iadd__(self, vec):
        self._checkType(vec)
        self._design += vec.GetDesign()
        self._dual += vec.GetDual()
        return self
            
    def __isub__(self, vec):
        self._checkType(vec)
        self._design -= vec.GetDesign()
        self._dual -= vec.GetDual()
        return self
        
    def __imul__(self, val):
        if isinstance(val, float):
            self._design *= val
            self._dual *= val
            return self
        else:
            raise TypeError('ReducedKKTVector.__imul__() >> ' + \
                            'Wrong argument type. Must be FLOAT.')
                            
    def __idiv__(self, val):
        self._checkType(vec)
        self._design /= val
        self._dual /= val
        return self
    
    def Equals(self, vec):
        self._checkType(vec)
        self._design = vec.GetDesign()
        self._dual = vec.GetDual()
        
    def EqualsAXPlusBY(self, a, x, b, y):
        self._checkType(x)
        self._checkType(y)
        self._design.EqualsAXPlusBY(a, x.GetDesign(), b, y.GetDesign())
        self._dual.EqualsAXPlusBY(a, x.GetDual(), b, y.GetDual())
        
    def Norm2(self):
        prodDesign = self._memory.InnerProd(self._design, self._design)
        prodDual = self._memory.InnerProd(self._dual, self._dual)
        totalProd = prodDesign + prodDual
        if totalProd < 0:
            raise ValueError('ReducedKKTVector.Norm2() >> ' + \
                             'Inner product is negative!')
        else:
            return sqrt(totalProd)
            
    def EqualsInitialGuess(self):
        self._design.EqualsInitialDesign()
        self._dual = 0.0
        
    def SetDesign(self, designVec):
        self._design = designVec
        
    def SetDual(self, dualVec):
        self._dual = dualVec
        
    def GetDesign(self):
        return self._design
        
    def GetDual(self):
        return self._dual