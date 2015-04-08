import numpy
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.composite import Hessian, ReducedKKTMatrix

class VectorFactory(object):
    """
    A factory object used for generating Kona's abstracted vector classes.

    This object also tallies up how many vectors of which kind needs to be
    allocated, based on the memory requirements of each optimization function.

    Attributes
    ----------
    _count : int
        Number of vectors required by optimization functions.
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _vec_type : PrimalVector, StateVector, DualVector
        Kona abstracted vector type associated with this factory

    Parameters
    ----------
    memory : KonaMemory
    vec_type : PrimalVector, StateVector, DualVector
    """

    def __init__(self, memory, vec_type=None):
        self.num_vecs = 0
        self._memory = memory
        if vec_type not in self._memory.vector_stack.keys():
            raise TypeError('VectorFactory() >> Unknown vector type!')
        else:
            self._vec_type = vec_type

    def request_num_vectors(self, count):
        if count < 1:
            raise ValueError('VectorFactory() >> Cannot request less than 1 vector.')
        self.num_vecs += count

    def generate(self):
        if self._memory.is_allocated:
            data = self._memory.pop_vector(self._vec_type)
            return self._vec_type(self._memory, data)
        else:
            raise RuntimeError('Can not generate vectors before memory allocation has happened')

            
class KonaMemory(object):
    """
    All-knowing Big Brother abstraction layer for Kona.

    Attributes
    ----------
    user_obj : UserSolver or derivative
        A user-defined solver object that implements specific elementary tasks.
    design_factory, state_factory, dual_factory: VectorFactory
        Factory objects for generating Kona's abstracted vector classes.
    precond_count : int
        Counter for tracking optimization cost.
    vector_stack : dict
        Memory stack for unused vector data.
    rank : int
        Processor rank.

    Parameters
    ----------
    user_obj : UserSolver or derivative
        A user-defined solver object that implements specific elementary tasks.
    """

    def __init__(self, user_obj=None):
        # assign user object
        self.user_obj = user_obj
        self.rank = self.user_obj.get_rank()

        # allocate vec assignments
        self.vector_stack = {
            PrimalVector : [],
            StateVector : [],
            DualVector : [],
        }

        # prepare vector factories
        self.primal_factory = VectorFactory(self, PrimalVector)
        self.state_factory = VectorFactory(self, StateVector)
        self.dual_factory = VectorFactory(self, DualVector)

        # cost tracking
        self.precond_count = 0

        self.is_allocated = False


    def push_vector(self, vec_type, user_data):
        """
        Pushes an unused user vector data container into the memory stack so it
        can be used later in a new vector.

        Parameters
        ----------
        vec_type : PrimalVector, StateVector, DualVector
            Vector type of the memory stack.
        user_data : BaseVector or derivative
            Unused user vector data container.
        """
        if user_data not in self.vector_stack[vec_type]:
            self.vector_stack[vec_type].append(user_data)

    def pop_vector(self, vec_type):
        """
        Take an unused user vector object out of the memory stack and serve it
        to the vector factory.

        Parameters
        ----------
        vec_type : PrimalVector, StateVector, DualVector
            Vector type to be popped from the stack.

        Returns
        -------
        BaseVector or derivative : user-defined vector data structure
        """
        if vec_type not in self.vector_stack.keys():
            raise TypeError('KonaMemory.pop_vector() >> ' + \
                            'Unknown vector type!')
        else:
            return self.vector_stack[vec_type].pop()

    def allocate_memory(self):
        """
        Absolute final stage of memory allocation.

        Once the number of required vectors are tallied up inside vector
        factories, this function will manipulate the user-defined solver object
        to allocate all actual, real memory required for the optimization.
        """

        if self.is_allocated:
            raise RuntimeError('Memory allready allocated, can-not re-allocate')

        allocator = self.user_obj.allocator

        self.vector_stack[PrimalVector] = allocator.alloc_primal(self.primal_factory.num_vecs)
        self.vector_stack[StateVector] = allocator.alloc_state(self.state_factory.num_vecs)
        self.vector_stack[DualVector] = allocator.alloc_dual(self.dual_factory.num_vecs)

        self.is_allocated = True
