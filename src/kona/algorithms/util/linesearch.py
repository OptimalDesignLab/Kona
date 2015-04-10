import sys

from kona.options import get_opt

class ILineSearch(object):
    pass

class StrongWolfe(object):
    pass

class BackTracking(object):

    def __init__(self, optns={}, out_file=sys.stdout):
        self.alpha_init = get_opt(optns, 1.0, 'alpha_init')

        self.alpha_min = get_opt(optns, 1e-4, 'alpha_min')

        self.rdtn_factor = get_opt(optns, .3, 'rdtn_factor')

        self.decr_cond = get_opt(optns, 5e-1, 'decr_cond')

        self.max_iter = get_opt(optns, 10, 'max_iter')
        self.p_dot_dfdx = get_opt(optns, 0.0, 'p_dot_dfdx')

        self.merit_function = None

    def find_step_length(self):

        if not (0 < self.alpha_init <= 1):
            raise ValueError('alpha_init must be 0 < alpha_init <=1')

        if self.p_dot_dfdx > 0.0:
            raise ValueError('search direction is not a descent direction')

        if not self.merit_function:
            raise ValueError('merit_function can not be None')

        merit = self.merit_function

        alpha = self.alpha_init
        f_init = merit(alpha)[0]
        f = f_init

        n_iter = 0
        while (alpha > self.alpha_min) and (n_iter < self.max_iter):
            if f <= f_init + self.decr_cond * alpha * self.p_dot_dfdx:
                return alpha, n_iter
            alpha *= self.rdtn_factor
            n_iter += 1
            f = merit(alpha)[0]

        raise Exception('LineSearch(run): failed to find a step length')
