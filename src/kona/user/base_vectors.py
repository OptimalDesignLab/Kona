import numpy as np

class BaseVector(object):

    def __init__(self, size, val=0):
        if np.isscalar(val):
            if val == 0:
                self.data = np.zeros(size, dtype=float)
            elif isinstance(val, (np.float, np.int)):
                self.data = np.ones(size, dtype=float)*val
        elif isinstance(val, (np.ndarray, list, tuple)):
            if size != len(val):
                raise ValueError('size given as %d, but length of value %d' % (size, len(val)))
            self.data = np.array(val)
        else:
            raise ValueError('val must be a scalar or array like, but was given as type %s' % (type(val)))

    def plus(self, vector):
        self.data += vector.data

    def times(self, value):
        self.data *= value

    def equals_value(self, value):
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
        self.data = a*x.data + b*y.data

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

    def __init__(self, num_primal, num_state, num_ceq):
        self.num_primal = num_primal
        self.num_state = num_state
        self.num_dual = num_ceq

    def alloc_primal(self, count):
        out = []
        for i in xrange(count):
            out.append(BaseVector(self.num_primal))
        return out

    def alloc_state(self, count):
        out = []
        for i in xrange(count):
            out.append(BaseVector(self.num_state))
        return out

    def alloc_dual(self, count):
        out = []
        for i in xrange(count):
            out.append(BaseVector(self.num_dual))
        return out
