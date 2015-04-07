import numpy as np

class BaseVector(object):

    def __init__(self, size, val=0):
        if np.isscalar(val):
            if val == 0:
                self.data = np.zeros(size)
            else:
                self.data = np.ones(size)*val
        elif isinstance(val, (np.ndarray, list, tuple)):
            if size != len(val):
                raise ValueError('size given as %d, but length of value %d' % (size, len(val)))
            self.data = np.array(val)
        else:
            raise ValueError('val must be a scalar or array like')


    def plus(self, vector):
        self.data += vector.data

    def times(self, value):
        self.data *= value.data

    def equals_value(self, val):
        self.data[:] = value

    def equals_vector(self, vector):
        self.data[:] = vector.data[:]

    def equals_ax_p_by(self, a, x, b, y):
        """
        User-defined linear algebra method for scalar multiplication and
        vector addition.

        .. math:: a\mathbf{x} + b\mathbf{y}

        Parameters
        ----------
        a, b : double
            Multiplication coefficients.
        x, y : numpy.ndarray
            Vectors to be operated on.
        out : numpy.ndarray
            Result of the operation.
        """
        self.data = a*x + b*y

    def inner(self, vector):
        """
        User-defined linear algebra method for a vector inner product.

        Parameters
        ----------
        x, y : numpy.ndarray
            Vectors to be operated on.

        Returns
        -------
        float : Result of the operation.
        """
        return np.inner(self.data, vector.data)

class BaseAllocator(object):

    def __init__(self, num_design, num_state, num_ceq):
        self.num_design = num_design
        self.num_state = num_state
        self.num_dual = num_ceq

    def alloc_design(self):
        return BaseVector(self.num_design)

    def alloc_state(self):
        return BaseVector(self.num_state)

    def alloc_dual(self):
        return BaseVector(self.num_dual)

class BaseAllocatorIDF(BaseAllocator):

    def __init__(self, num_design, num_state, num_ceq):
        self.num_real_design = num_design
        self.num_real_ceq = num_ceq
        super(BaseAllocatorIDF, self).__init__(self.num_real_design + self.num_state,
                              self.num_state,
                              self.num_real_ceq + self.num_state)

    def restrict_design(self, opType, target):
        """
        If operation type is 0 (``type == 0``), set the target state variables
        to zero.

        If operation type is 1 (``type == 1``), set the real design variables
        to zero.

        Parameters
        ----------
        opType : int
            Operation type flag.
        vec : numpy.ndarray
            Design vector to be operated on.
        """
        if opType == 0:
            vec[self.num_real_design:] = 0.
        elif opType == 1:
            vec[:self.num_real_design] = 0.
        else:
            raise ValueError('Unexpected type in restrict_design()!')

    def copy_dual_to_targstate(self, takeFrom, copyTo):
        """
        Take the target state variables from dual storage and put them into
        design storage. Also set the real design variables to zero.

        Parameters
        ----------
        take_from : numpy.ndarray
            Vector from where target state variables should be taken.
        copy_to : numpy.ndarray
            Vector to which target state variables should be copied.
        """
        copyTo[:self.num_real_design] = 0.
        copyTo[self.num_real_design:] = takeFrom[self.num_real_ceq:]

    def copy_targstate_to_dual(self, take_from, copy_to):
        """
        Take the target state variables from design storage and put them into
        dual storage.

        Parameters
        ----------
        take_from : int
            Design index from where target state variables should be taken.
        copy_to : int
            Dual index to which target state variables should be copied.
        """
        copyTo[self.num_real_ceq:] = takeFrom[self.num_real_design:]
