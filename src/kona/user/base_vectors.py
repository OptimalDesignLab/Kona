import numpy as np

class BaseVector(object):
    """
    Kona's default data container, implemented on top of NumPy arrays.

    Any user defined data container must implement all the methods below.

    These vectors are initialized by the user-created `BaseAllocator` object.
    Therefore, the initialization implementation does not need to exactly
    follow the example below. The user is free to initialize these vector
    objects any which way they like, as long as it is in sync with the
    `BaseAllocator` implementation.

    Parameters
    ----------
    size: int
        Size of the 1-D numpy vector contained in this object.
    val : float or array-like, optional
        Data value for vector initialization.

    Attributes
    ----------
    data : numpy.array
        Numpy vector containing numerical data.
    """
    def __init__(self, size, val=0):
        if np.isscalar(val):
            if val == 0:
                self.data = np.zeros(size, dtype=float)
            elif isinstance(val, (np.float, np.int)):
                self.data = np.ones(size, dtype=float)*val
        elif isinstance(val, (np.ndarray, list, tuple)):
            if size != len(val):
                raise ValueError(
                    'size given as %d, but length of value %d'%(size, len(val)))
            self.data = np.array(val)
        else:
            raise ValueError(
                'val must be a scalar or array like, ' +
                'but was given as type %s'%(type(val)))

    def plus(self, vector):
        """
        Add the given vector to this vector.

        Parameters
        ----------
        vector : BaseVector
            Incoming vector for in-place operation.
        """
        self.data = self.data + vector.data

    def times_scalar(self, value):
        """
        Multiply all elements of this vector with the given scalar.

        Parameters
        ----------
        value: float
        """
        self.data = value*self.data

    def times_vector(self, vector):
        """
        Perform element-wise multiplication between vectors.

        Parameters
        ----------
        vector : BaseVector
            Incoming vector for in-place operation.
        """
        self.data = self.data*vector.data

    def equals_value(self, value):
        """
        Set all elements of this vector to given scalar value.

        Parameters
        ----------
        value : float
        """
        self.data[:] = value

    def equals_vector(self, vector):
        """
        Set this vector equal to the given vector.

        Parameters
        ----------
        vector : BaseVector
            Incoming vector for in-place operation.
        """
        self.data[:] = vector.data[:]

    def equals_ax_p_by(self, a, x, b, y):
        """
        Perform the elementwise scaled addition defined below:

        .. math:: a\\mathbf{x} + b\\mathbf{y}

        The result is saved into this vector.

        Parameters
        ----------
        a : double
            Scalar coefficient of ``x``.
        x : BaseVector
            Vector to be operated on.
        b : double
            Scalar coefficient of ``y``.
        y : BaseVector
            Vector to be operated on.
        """
        self.data = a*x.data + b*y.data

    def inner(self, vector):
        """
        Perform an inner product between the given vector and this one.

        Parameters
        ----------
        vector : BaseVector
            Incoming vector for in-place operation.

        Returns
        -------
        float
            Result of the operation.
        """
        if len(self.data) == 0:
            return 0.
        else:
            return np.inner(self.data, vector.data)

    @property
    def infty(self):
        """
        Infinity norm of the vector.

        Returns
        -------
        float
            Infinity norm.
        """
        if len(self.data) == 0:
            return 0.
        else:
            return np.linalg.norm(self.data, np.inf)

    def exp(self, vector):
        """
        Calculate element-wise exponential operation on the vector.

        Parameters
        ----------
        vector : BaseVector
            Incoming vector for in-place operation.
        """
        self.data = np.exp(vector.data)

    def log(self, vector):
        """
        Calculate element-wise natural log operation on the vector.

        Kona will never call this on zero-valued vectors. No special handling
        of zero values necessary.

        Parameters
        ----------
        vector : BaseVector
            Incoming vector for in-place operation.
        """
        self.data = np.log(vector.data)

    def pow(self, power):
        """
        Calculate element-wise power operation on the vector.

        Kona will never call a negative power on zero-valued vectors. No
        special handling of zero values necessary.

        Parameters
        ----------
        power : float
        """
        self.data = self.data**power
