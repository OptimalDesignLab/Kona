import numpy
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector

class KonaMatrix(object):

    def __init__(self, solver):
        self._solver = solver

    def linearize(self, primal, state):
        self._primal = primal
        self._state = state

    def _check_type(self, vector, reference):
        if not isinstance(vector, reference):
            raise TypeError('KonaMatrix() >> ' + \
                            'Wrong vector type. Must be a %s.' % reference)

    def product(self, in_vec, out_vec):
        pass

    @property
    def tranpose(self):
        pass

class dRdX(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_type(in_vec, PrimalVector)
        self._check_type(out_vec, StateVector)
        self._solver.multiply_jac_d(self._primal._data, self._state._data,
                                    in_vec._data, out_vec._data)

class dRdU(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_type(in_vec, StateVector)
        self._check_type(out_vec, StateVector)
        self._solver.multiply_jac_s(self._primal._data, self._state._data,
                                    in_vec._data, out_vec._data)

class dCdX(KonaMatrix):

    def product(self, in_vec, out_vec):
        self._check_type(in_vec, PrimalVector)
        self._check_type(out_vec, DualVector)
        self._solver.multiply_ceqjac_d(self._primal._data, self._state._data,
                                    in_vec._data, out_vec._data)

class dCdU(KonaMatrix):
    def product(self, in_vec, out_vec):
        self._check_type(in_vec, StateVector)
        self._check_type(out_vec, DualVector)
        self._solver.multiply_ceqjac_s(self._primal._data, self._state._data,
                                    in_vec._data, out_vec._data)
