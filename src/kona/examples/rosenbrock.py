import numpy

from kona.user import UserSolver
from kona.user import BaseVector

class Rosenbrock(UserSolver):

    def __init__(self):
        super(rosenbrock, self).__init__(2,0,0)

    def eval_obj(self, at_design, at_state): 
        x = at_design.data[0]
        y = at_design.data[1]
        obj = (1.0 - x)*(1.0 - x) + 100.0*(y - x*x)*(y - x*x)
        precond_calls = 0
        return (obj, precond_calls)

    def eval_dFdX(self, at_design, at_state, store_here):
        x = at_design.data[0]
        y = at_design.data[1]
        dFdx1 = -2.0*(1.0 - x) - 400.0*x*(y - x*x)
        dFdx2 = 200.0*(y - x*x)
        store_here.data = numpy.array([dFdx1, dFdx2])

    def init_design(self, store_here):
        store_here.data = numpy.array([5., 5.])













