import numpy

from kona.user import UserSolver

class Rosenbrock(UserSolver):

    def __init__(self, ndv):
        super(Rosenbrock, self).__init__(
            num_design=dv,
            num_state=0,
            num_eq=0,
            num_ineq=0)

    def eval_obj(self, at_design, at_state):
        x = at_design
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    def eval_dFdX(self, at_design, at_state):
        x = at_design
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = numpy.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der

    def init_design(self):
        return -numpy.ones(self.num_design)
