import numpy
from kona.common import Singleton
from kona.vectors.common import DesignVector, StateVector, DualVector
from kona.vectors.composite import *

class VectorFactory(object):

    def __init__(self, memory, vec_type=None):
        self._count = 0
        self._memory = memory
        self._vec_type = vec_type

    def tally(self, number):
        self._count += number

    def generate(self):
        data = self._memory.pop_vector(self._vec_type)
        return self._vec_type(self._memory, data)

class KonaMemory(Singleton):

    def __init__(self, user_obj):
        # assign user object
        self.user_obj = user_obj

        # prepare vector factories
        self.design_factory = VectorFactory(self, DesignVector)
        self.state_factory = VectorFactory(self, StateVector)
        self.dual_factory = VectorFactory(self, DualVector)

        # cost tracking
        self.precond_count = 0

        # allocate vec assignments
        self.vector_stack = {
            DesignVector : [],
            StateVector : [],
            DualVector : [],
        }

    def push_vector(self, vec_type, user_data):
        self.vector_stack[vec_type].append(user_data)

    def pop_vector(self, vec_type):
        return self.vector_stack[vec_type].pop()

    def allocate_memory(self):
        self.vector_stack[DesignVector] = \
            self.user_obj.allocator.alloc_design(self.design_factory._count)
        self.vector_stack[StateVector] = \
            self.user_obj.allocator.alloc_state(self.state_factory._count)
        self.vector_stack[DualVector] = \
            self.user_obj.allocator.alloc_dual(self.dual_factory._count)

    def get_rank():
        return self.user_obj.get_rank()
