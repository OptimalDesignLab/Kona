from numpy import sqrt
from kona.linalg.vectors.common import DesignVector, StateVector, DualVector

class ReducedKKTVector(object):
    """
    A composite vector representing a combined design and dual vectors.
    
    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _design : DesignVector
        Design component of the composite vector.
    _dual : DualVector
        Dual components of the composite vector.
        
    Parameters
    ----------
    memory: KonaMemory
    design_vec : DesignVector
    dual_vec : DualVector
    """
    def __init__(self, memory, design_vec, dual_vec):
        self._memory = memory
        if isinstance(designVec, DesignVector):
            self._design = design_vec
        else:
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Unidentified design vector.')
        if isinstance(dualVec, DualVector):
            self._dual = dual_vec
        else:
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Unidentified dual vector.')
                            
    def _checkType(self, vec):
        if not isinstance(vec, type(self)):
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Wrong vector type. Must be %s' % type(self))
                            
    def equals(self, vector):
        self._check_type(vector)
        self._design.equals(vector._design)
        self._dual.equals(vector._dual)
                            
    def plus(self, vector):
        self._check_type(vector)
        self._design.plus(vector._design)
        self._dual.plus(vector._dual)
            
    def minus(self, vector):
        self._check_type(vector)
        self._design.minus(vector._design)
        self._dual.minus(vector._dual)
        
    def times(self, value):
        if isinstance(value, float):
            self._design.times(value)
            self._dual.times(value)
        else:
            raise TypeError('ReducedKKTVector.times() >> ' + \
                            'Wrong argument type. Must be FLOAT.')
                            
    def divide_by(self, value):
        self.times(1./value)
        
    def equals_ax_p_by(self, a, x, b, y):
        self._check_type(x)
        self._check_type(y)
        self._design.equals_ax_p_by(a, x._design, b, y_design)
        self._dual.equals_ax_p_by(a, x._dual, b, y_dual)
        
    def inner(self, vector):
        self._check_type(vector)
        design_prod = self._design.inner(vector._design)
        dual_prod = self._dual.inner(vector._dual)
        return design_prod + dual_prod
    
    @property
    def norm2(self):
        prod = self.inner(self)
        if prod < 0:
            raise ValueError('ReducedKKTVector.norm2 >> ' + \
                             'Inner product is negative!')
        else:
            return sqrt(prod)
            
    def equals_initial_guess(self):
        self._design.equals_initial_design()
        self._dual.equals(0.0)