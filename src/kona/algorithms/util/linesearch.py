class LineSearch(object):
    pass

class StrongWolfe(LineSearch):
    pass

class BackTracking(LineSearch):

    def __init__(self):
        self.alpha_init = 1.0
        self.alpha_min = 1e-4
        self.rdtn_factor = 0.3
        self.decr_cond = 5e-1
        self.curv_cond = 5e-1
        self.max_iter = 10
        self.p_dot_dfdx_ = 0.0

    def set_initial_step_length(self, alpha_init):
        assert 0. < alpha_init <= 1.
        self.alpha_init = alpha_init

    def set_minimum_step(self, alpha_min):
        self.alpha_min = alpha_min

    def set_rdtn_factor(self, rdtn_factor):
        self.rdtn_factor = rdtn_factor

    def set_conv(self, decr_cond, curv_cond, max_iter):
        self.decr_cond = decr_cond
        self.curv_cond = curv_cond
        self.max_iter = max_iter    

    def set_search_dot_grad(self, p_dot_grad):
        self.p_dot_dfdx_ = p_dot_grad
        if self.p_dot_dfdx_ > 0.0:
            raise Exception('LineSearch(SetSearchDotGrad):' +
                            'search direction is not a descent direction')

    def set_merit_function(self, merit_function):
        self.merit_function = merit_function

    def find_step_length(self):
        func = self.merit_function.eval_func

        alpha = 1.0
        f_init = func(0.0)
        f = func(alpha)

        counter = 0
        while counter < max_iter:
            if f <= f_init + self.decr_cond * alpha * self.p_dot_dfdx_:
                return alpha
            else:
                alpha *= self.rdtn_factor
                f = func(alpha)

        raise Exception('LineSearch(run): failed to find a step length')
