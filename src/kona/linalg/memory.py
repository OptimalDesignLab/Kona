import numpy
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector

class VectorFactory(object):
    """
    A factory object used for generating Kona's abstracted vector classes.

    This object also tallies up how many vectors of which kind needs to be
    allocated, based on the memory requirements of each optimization function.

    Parameters
    ----------
    memory : KonaMemory
    vec_type : PrimalVector or StateVector or DualVector

    Attributes
    ----------
    num_vecs : int
        Number of vectors requested from this factory.
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _vec_type : PrimalVector or StateVector or DualVector
        Kona abstracted vector type associated with this factory
    """

    def __init__(self, memory, vec_type=None):
        self.num_vecs = 0
        self._memory = memory
        if vec_type not in self._memory.vector_stack.keys():
            raise TypeError('VectorFactory() >> Unknown vector type!')
        else:
            self._vec_type = vec_type

    def request_num_vectors(self, count):
        """
        Put in a request for the factory's vector type, to be used later.

        Parameters
        ----------
        count : int
            Number of vectors requested.
        """
        if count < 1:
            raise ValueError('VectorFactory() >> ' + \
                             'Cannot request less than 1 vector.')
        self.num_vecs += count

    def generate(self):
        """
        Generate one abstract KonaVector of this vector factory's defined type.

        Returns
        -------
        KonaVector
            Abstracted vector type linked to user generated memory.
        """
        if self._memory.is_allocated:
            try:
                data = self._memory.pop_vector(self._vec_type)
            except IndexError:
                raise MemoryError('no more vector memory available. Allocate more vectors in your algorithm initialization')
            return self._vec_type(self._memory, data)
        else:
            raise RuntimeError('VectorFactory() >> ' + \
                               'Must allocate memory before generating vector.')

class KonaMemory(object):
    """
    All-knowing Big Brother abstraction layer for Kona.

    Parameters
    ----------
    solver : UserSolver
        A user-defined solver object that implements specific elementary tasks.

    Attributes
    ----------
    solver : UserSolver
        A user-defined solver object that implements specific elementary tasks.
    primal_factory : VectorFactory
        Vector generator for primal space.
    state_factory : VectorFactory
        Vector generator for state space.
    dual_factory : VectorFactory
        Vector generatorfor dual space.
    precond_count : int
        Counter for tracking optimization cost.
    vector_stack : dict
        Memory stack for unused vector data.
    rank : int
        Processor rank.
    """

    def __init__(self, solver):
        # assign user object
        self.solver = solver
        self.rank = self.solver.get_rank()

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
        vec_type : KonaVector
            Vector type of the memory stack.
        user_data : BaseVector
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
        vec_type : KonaVector
            Vector type to be popped from the stack.

        Returns
        -------
        BaseVector
            User-defined vector data structure.
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

        allocator = self.solver.allocator

        self.vector_stack[PrimalVector] = allocator.alloc_primal(self.primal_factory.num_vecs)
        self.vector_stack[StateVector] = allocator.alloc_state(self.state_factory.num_vecs)
        self.vector_stack[DualVector] = allocator.alloc_dual(self.dual_factory.num_vecs)

        self.is_allocated = True
