import numpy

from kona.user import UserSolver
from kona.user import BaseVector

class Rosenbrock(UserSolver):

    def __init__(self, num_design):
        super(Rosenbrock, self).__init__(num_design,0,0)

    def eval_obj(self, at_design, at_state):
        x = at_design.data
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    def eval_dFdX(self, at_design, at_state, store_here):
        x = at_design.data
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = numpy.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        store_here.data[:] = der[:]

    def init_design(self, store_here):
        store_here.data[:] = -1.
