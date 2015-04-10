from kona.options import get_opt

class ILineSearch(object):
    pass

class StrongWolfe(object):
    pass

class BackTracking(object):

    def __init__(self, optns={}):
        opt_alpha_init = get_opt(optns, 'alpha_init')
        self.alpha_init = get_opt(optns, 1.0, 'alpha_init')

        opt_alpha_min = get_opt(optns, 'alpha_min')
        self.alpha_min = opt_alpha_min if opt_alpha_min else 1e-4

        opt_rdtn_factor = get_opt(optns, 'rdtn_factor')
        self.rdtn_factor = opt_rdtn_factor if opt_rdtn_factor else 0.3

        opt_decr_cond = get_opt(optns, 'decr_cond')
        self.decr_cond = opt_decr_cond if opt_decr_cond else 5e-1
        self.max_iter = 10
        self.p_dot_dfdx = 0.0
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
        f_init = merit(alpha)
        f = f_init

        n_iter = 0
        while (alpha > self.alpha_min) and (n_iter < self.max_iter):
            if f <= f_init + self.decr_cond * alpha * self.p_dot_dfdx:
                return alpha, n_iter
            alpha *= self.rdtn_factor
            n_iter += 1
            f = merit(alpha)

        raise Exception('LineSearch(run): failed to find a step length')
