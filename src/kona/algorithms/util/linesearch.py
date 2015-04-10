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

        if self.merit_function is None:
            raise ValueError('merit_function can not be None')

    def find_step_length(self):
        pass

class StrongWolfe(LineSearch):

    def __init__(self, optns={}, out=sys.stdout):
        super(BackTracking, self).__init__(optns, out)
        self.alpha_max = get_opt(optns, 1.0, 'alpha_max')
        self.curv_cond = get_opt(optns, 0.7, 'curv_cond')

    def _validate_options(self):
        super(StrongWolfe, self)._validate_options()

        if not (self.alpha_max > 0.) and not (self.alpha_max > self.alpha_init):
            raise ValueError('alpha_max must be > 0 and > alpha_init')

        if not (self.decr_cond < self.curv_cond < 1):
            raise ValueError('curv_cond must be suff_cond < curv_cond < 1')

    def zoom(self):
        pass

    def find_step_length(self):
        self._validate_options()

        merit = self.merit_function
        phi_init = merit.eval_func(0.0)
        dphi_init = merit.eval_grad(0.0)

        self.alpha_old = 0.0
        phi_old = phi_init
        self.dphi_old = dphi_init

        if self.dphi_old > 0.0:
            raise ValueError('search direction is not a descent direction')

        self.alpha_new = self.alpha_init
        self.phi_new = 0.0
        self.phi_new = 0.0
        quad_coeff = 0.0
        self.deriv_hi = False

        # START OF BIG FOR-LOOP
        for i in xrange(self.max_iter):

            # get new step
            if i == 0:
                self.alpha_new = self.alpha_init
            else:
                if quad_coeff > 0:
                    self.alpha_new = self.alpha_old - 0.5*self.dphi_old/quad_coeff
                    if (self.alpha_new < self.alpha_old) or (self.alpha_new > self.alpha_max):
                        self.alpha_new = min(2*self.alpha_old, self.alpha_max)
                else:
                    self.alpha_new = min(2*self.alpha_old, self.alpha_max)

            # get new function value
            self.phi_new = merit.eval_func(self.alpha_new)

            # if new step violates sufficient decrease, call zoom
            phi_sufficient = phi_init + decr_cond*self.alpha_new*dphi_init
            if (self.phi_new > phi_sufficient) or \
            ((i > 0) and (self.phi_new >= phi_old)):
                self.phi_new = 0.0
                self.deriv_hi = False
                return self.zoom()

            # otherwise, get new gradient
            self.phi_new = merit.eval_grad(self.alpha_new)

            # check curvature condition
            if abs(self.phi_new) <= -curv_cond*dphi_init:

                # if curvature condition is satisfied, return self.alpha_new
                if curv_cond > 1.e-6:
                    return self.alpha_new

                # a very small curvature is supicious, check for local minimum
                # perturb in one direction first
                perturb = merit.eval_func(self.alpha_new - self.alpha_max*1.e-6)

                # if the perturbation yielded a smaller function, update
                if perturb < self.phi_new:
                    self.phi_new = perturb
                    self.phi_new = merit.eval_func(
                        self.alpha_new - self.alpha_max*1.e-6)
                else:
                    # otherwise perturb in the other direction
                    perturb = merit.eval_func(
                        self.alpha_new + self.alpha_max*1.e-6)
                    # if the perturbation yielded a smaller function, update
                    if perturb < self.phi_new:
                        self.phi_new = perturb
                        self.phi_new = merit.eval_func(
                            self.alpha_new + self.alpha_max*1.e-6)
                    else:
                        # if neither perturbation worked, we have a true minimum
                        return self.alpha_new

            # check if new gradient is positive
            if self.phi_new >= 0:
                # if we get this far, the curvature condition is not satisfied
                self.deriv_hi = True
                return self.zoom

            # update old variables
            quad_coeff = self.alpha_new - self.alpha_old
            quad_coeff = ((self.phi_new - self.phi_old) -
                self.dphi_new*quad_coeff) / (quad_coeff**2)
            self.alpha_old = self.alpha_new
            self.phi_old = self.phi_new
            self.dphi_old = self.dphi_new

        # END OF BIG FOR-LOOP

        # if we got here then we didn't find a step
        raise Exception('LineSearch(run): failed to find a step length')

class BackTracking(LineSearch):

    def __init__(self, optns={}, out=sys.stdout):
        super(BackTracking, self).__init__(optns, out=sys.stdout)
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

        # if we get here, we didn't find a step
        raise Exception('LineSearch(run): failed to find a step length')
