class ILineSearch(object):
    pass

class StrongWolfe(object):
    pass

class BackTracking(object):

    def __init__(self):
        self.alpha_init = 1.0
        self.alpha_min = 1e-4
        self.rdtn_factor = 0.3
        self.decr_cond = 5e-1
        self.curv_cond = 5e-1
        self.max_iter = 10
        self.p_dot_dfdx = 0.0
        self.merit_function = None

    def find_step_length(self):

        if not (0 < self.alpha_init <= 1):
            raise ValueError('alpha_init must be 0 < alpha_init <=1')

        if self.p_dot_dfdx > 0.0:
            raise Exception('LineSearch(SetSearchDotGrad):' +
                            'search direction is not a descent direction')

        if not self.merit_function:
            raise ValueError('merit_function can not be None')

        merit = self.merit_function

        alpha = self.alpha_init
        f_init = merit(alpha)
        f = f_init

        # for i in xrange(self.max_iter):
        #     print "foo", i, alpha, f, f_init + self.decr_cond * alpha * self.p_dot_dfdx
        #     if f <= f_init + self.decr_cond * alpha * self.p_dot_dfdx:
        #         return alpha, i+1
        #     else:
        #         alpha *= self.rdtn_factor
        #         f = func(alpha)

        n_iter = 0
        while (alpha > self.alpha_min) and (n_iter < self.max_iter):
            if f <= f_init + self.decr_cond * alpha * self.p_dot_dfdx:
                return alpha, n_iter
            alpha *= self.rdtn_factor
            n_iter += 1
            f = merit(alpha)

        raise Exception('LineSearch(run): failed to find a step length')
