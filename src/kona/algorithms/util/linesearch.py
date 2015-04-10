import sys

from kona.options import get_opt

class LineSearch(object):

    def __init__(self, optns={}, out=sys.stdout):
        self.alpha_init = get_opt(optns, 1.0, 'alpha_init')
        self.decr_cond = get_opt(optns, 5e-1, 'decr_cond')
        self.max_iter = get_opt(optns, 10, 'max_iter')
        self.merit_function = None

    def _validate_options(self):
        if not (self.max_iter >= 1):
            raise ValueError('max_iter must be a positive integer')

        if not (0 < self.alpha_init <= 1):
            raise ValueError('alpha_init must be 0 < alpha_init <=1')

        if not (0 < self.decr_cond < 1):
            raise ValueError('suff_cond must be 0 < suff_cond < 1')

        if not self.merit_function:
            raise ValueError('merit_function can not be None')

    def find_step_length(self):
        pass

class StrongWolfe(LineSearch):

    def __init__(self, optns={}):
        super(BackTracking, self).__init__(optns)
        self.alpha_max = get_opt(optns, 1.0, 'alpha_max')
        self.curv_cond = get_opt(optns, 0.7, 'curv_cond')

    def _validate_options(self):
        super(StrongWolfe, self)._validate_options()

        if not (self.alpha_max > 0.) and not (self.alpha_max > self.alpha_init):
            raise ValueError('alpha_max must be > 0 and > alpha_init')

        if not (self.decr_cond < self.curv_cond < 1):
            raise ValueError('curv_cond must be suff_cond < curv_cond < 1')

    def zoom(self):
        self.max_iter = get_opt(optns, 10, 'max_iter')
        self.p_dot_dfdx = get_opt(optns, 0.0, 'p_dot_dfdx')

        self.merit_function = None

    def find_step_length(self):
        self._validate_options()

        merit = self.merit_function
        phi_init = merit.eval_func(0.0)
        dphi_init = merit.eval_grad(0.0)

        alpha_old = 0.0
        phi_old = phi_init
        dphi_old = dphi_init

        if dphi_old > 0.0:
            raise ValueError('search direction is not a descent direction')

        alpha_new = 0.0
        phi_new = 0.0
        dphi_new = 0.0
        quad_coeff = 0.0
        deriv_hi = False

        for i in xrange(self.max_iter):
            # get new step
            if i == 0:
                alpha_new = self.alpha_init
            else:
                if quad_coeff > 0:
                    alpha_new = alpha_old - 0.5*dphi_old/quad_coeff
                    if (alpha_new < alpha_old) or (alpha_new > self.alpha_max):
                        alpha_new = min(2*alpha_old, alpha_max)
                else:
                    alpha_new = min(2*alpha_old, alpha_max)
            # get new function value
            phi_new = merit.eval_func(alpha_new)
            # if new step violates sufficient decrease, call zoom
            if (phi_new > phi_init + decr_cond*alpha_new*dphi_init) or \
                ( (i > 0) and (phi_new >= phi_old) ):
                dphi_new = 0.0
                deriv_hi = False
                return self.zoom()
            # otherwise, get new gradient
            dphi_new = merit.eval_grad(alpha_new)
            # check curvature condition
            if abs(dphi_new) <= -curv_cond*dphi_init:
                if curv_cond > 1.e-6:
                    return alpha_new




class BackTracking(LineSearch):

    def __init__(self, optns={}):
        super(BackTracking, self).__init__(optns)
        self.alpha_min = get_opt(optns, 1e-4, 'alpha_min')
        self.rdtn_factor = get_opt(optns, .3, 'rdtn_factor')
        self.p_dot_dfdx = 0.0

    def _validate_options(self):
        super(BackTracking, self)._validate_options()

        if self.p_dot_dfdx > 0.0:
            raise ValueError('search direction is not a descent direction')

        if not (0 <= self.rdtn_factor <= 1):
            raise ValueError('reduction factor must be 0 <= rdtn_factor <= 1')

    def find_step_length(self):
        self._validate_options()

        merit = self.merit_function
        alpha = self.alpha_init

        f_init = merit.eval_func(alpha)
        f = f_init

        n_iter = 0
        while (alpha > self.alpha_min) and (n_iter < self.max_iter):
            if f <= f_init + self.decr_cond * alpha * self.p_dot_dfdx:
                return alpha, n_iter
            alpha *= self.rdtn_factor
            n_iter += 1

            f = merit.eval_func(alpha)

        raise Exception('LineSearch(run): failed to find a step length')
