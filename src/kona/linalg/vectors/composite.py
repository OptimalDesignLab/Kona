import numpy as np
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector

class ReducedKKTVector(object):
    """
    A composite vector representing a combined design and dual vectors.

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _design : PrimalVector
        Design component of the composite vector.
    _dual : DualVector
        Dual components of the composite vector.

    Parameters
    ----------
    memory: KonaMemory
    primal_vec : PrimalVector
    dual_vec : DualVector
    """
    def __init__(self, memory, primal_vec, dual_vec):
        self._memory = memory
        if isinstance(primal_vec, PrimalVector):
            self._primal = primal_vec
        else:
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Unidentified design vector.')
        if isinstance(dual_vec, DualVector):
            self._dual = dual_vec
        else:
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Unidentified dual vector.')

    def _check_type(self, vec):
        if not isinstance(vec, type(self)):
            raise TypeError('ReducedKKTVector() >> ' + \
                            'Wrong vector type. Must be %s' % type(self))

    def equals(self, vector):
        self._check_type(vector)
        self._primal.equals(vector._primal)
        self._dual.equals(vector._dual)

    def plus(self, vector):
        self._check_type(vector)
        self._primal.plus(vector._primal)
        self._dual.plus(vector._dual)

    def minus(self, vector):
        self._check_type(vector)
        self._primal.minus(vector._primal)
        self._dual.minus(vector._dual)

    def times(self, value):
        if isinstance(value, (float, int, np.float64, np.int64, np.float32, np.int32)):
            self._primal.times(value)
            self._dual.times(value)
        else:
            raise TypeError('ReducedKKTVector.times() >> ' + \
                            'Wrong argument type. Must be FLOAT.')

    def divide_by(self, value):
        self.times(1./value)

    def equals_ax_p_by(self, a, x, b, y):
        self._check_type(x)
        self._check_type(y)
        self._primal.equals_ax_p_by(a, x._primal, b, y._primal)
        self._dual.equals_ax_p_by(a, x._dual, b, y._dual)

    def inner(self, vector):
        self._check_type(vector)
        primal_prod = self._primal.inner(vector._primal)
        dual_prod = self._dual.inner(vector._dual)
        return primal_prod + dual_prod

    @property
    def norm2(self):
        prod = self.inner(self)
        if prod < 0:
            raise ValueError('ReducedKKTVector.norm2 >> ' + \
                             'Inner product is negative!')
        else:
            return np.sqrt(prod)

    def equals_init_guess(self):
        self._primal.equals_init_design()
        self._dual.equals(0.0)
