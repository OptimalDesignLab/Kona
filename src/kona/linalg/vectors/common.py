from numpy import sqrt

class KonaVector(object):
    """
    An abstract vector class connected to the Kona user memory, containing a
    common set of algebraic member functions.
    
    Attributes
    ----------
    _memory : UserMemory (singleton)
        Pointer to the Kona user memory.
    _data : BaseVector or derivative
        User defined vector object that contains data and operations on data.
        
    Parameters
    ----------
    memory : UserMemory (singleton)
        Pointer to the Kona user memory.
    """
    _flag = None # vector basis flag -- 0 is design, 1 is state, 2 is dual
    
    def __init__(self, user_vector, memory_obj):
        self._memory = memory_obj
        self._data = user_vector
        
    def _check_type(self, vector):
        if not isinstance(vector, type(self)):
            raise TypeError('KonaVector() >> ' + \
                            'Vector type mismatch. Must be %s' % type(self))
                            
    def equals(self, rhs): # the = operator cannot be overloaded
        if isinstance(rhs, float):
            self._data.equals_value(rhs)
        else:
            self._check_type(rhs)
            self._data.equals_vector(rhs._data)
            
    def plus(self, vector): # this is the += operator
        self._check_type(vector)
        self._data.plus(vector._data)
        
    def minus(self, vector): # this is the -= operator
        self._check_type(vector)
        self._data.times(-1.)
        self._data.plus(vector._data)
        self._data.times(-1.)
        
    def times(self, value): # this is the *= operator
        if not isinstance(value, float):
            self._data.times(value)
        else:
            raise TypeError('KonaVector() >> ' + \
                            'Argument must be a float.')
        
    def divide_by(self, val): # this is the /= operator
        self.times(1./val)
        
    def equals_ax_p_b(self, a, x, b, y): # this performs self = a*x + b*y
        self._memory.CheckType(x, type(self))
        self._memory.CheckType(y, type(self))
        self._memory.AXPlusBY(a, x.GetIndex(), b, y.GetIndex(), self)
    
    @property
    def norm2(self): # this takes the L2 norm of the vector
        prod = self._data.inner(self._data)
        if prod < 0:
            raise ValueError('KonaVector.norm2 >> ' + \
                             'Inner product is negative!')
        else:
            return sqrt(prod)
    
class DesignVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific 
    to design vectors.
    """    
    def restrict_target_state(self):
        self._memory.allocator.restrict_design(0, self._data)
        
    def restrict_real_design(self):
        self._memory.solver.allocator.restrict_design(1, self._data)
        
    def convert(self, dual_vector):
        self._memory.allocator.copy_dual_to_targstate(dual_vector._data, 
                                                     self._data)
        
    def equals_init_design(self):
        self._memory.solver.init_design(self._data)
        
    def equals_objective_gradient(self, at_design, at_state):
        self._memory.solver.eval_obj_d(at_design._data,
                                       at_state._data,
                                       self._data)
        
    def equals_reduced_gradient(self, at_design, at_state, at_adjoint, work):
        pass
        
    def equals_lagrangian_reduced_grad(self, at_design, atstate, at_dual, 
                                       at_adjoint, work):
        pass
    
class StateVector(KonaVector):
    """
    Derived from the base abstracted vector. Contains member functions specific 
    to state vectors.
    """
    def equals_objective_partial(self, at_design, at_state):
        self._memory.solver.eval_obj_s(at_design._data,
                                       at_state._data,
                                       self._data)
    
    def equals_PDE_residual(self, at_design, at_state):
        self._memory.solver.eval_residual(at_design._data,
                                          at_state._data,
                                          self._data)
        
    def equals_primal_solution(self, at_design):
        self._memory.solver.solve_system(at_design._data, self._data)
        
class DualVector(BaseVector):
    """
    Derived from the base abstracted vector. Contains member functions specific 
    to state vectors.
    """
    
    def convert(self, design):
        self._memory.solver.copy_targstate_to_dual(design._data, self._data)
        
    def equals_constraints(self, at_design, at_state):
        self._memory.solver.eval_ceq(at_design._data,
                                     at_state._data,
                                     self._data)