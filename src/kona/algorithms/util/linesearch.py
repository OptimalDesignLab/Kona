import sys

from kona.options import get_opt

from kona.algorithms.util.merit import MeritFunction

################################################################################
#                        Step Interpolation Functions                          #
################################################################################

def quadratic_step(alpha_low, alpha_hi, f_low, f_hi, df_low):
    """
    Finds a new step between ``alpha_low`` and ``alpha_hi`` using quadratic
    interpolation.

    Parameters
    ----------
    alpha_low : float
        Lower bound for step calculations.
    alpha_hi : float
        Upper bound for step calculations.
    f_low : float
        Function value at lower bound.
    f_hi : float
        Function value at upper bound.
    df_low : float
        Function derivative at lower bound.

    Returns
    -------
    float : Interpolated step size.
    """
    dalpha = alpha_hi - alpha_low
    step = alpha_low - 0.5*df_low*(dalpha**2)/(f_hi - f_low - df_low*dalpha)

    min_alpha = min(alpha_hi, alpha_low)
    max_alpha = max(alpha_hi, alpha_low)
    if max_alpha < step < min_alpha:
        raise Exception('quadratic_step(): step = %f\n'%step +
                        'alpha_low = %f\n'%alpha_low +
                        'alpha_hi = %f\n'%alpha_hi +
                        'f_low = %f\n'%f_low +
                        'f_hi = %f\n'%f_hi +
                        'df_low = %f\n'%df_low +
                        '>> Check StrongWolfe._zoom() for bugs! <<')

    if (step - min_alpha) < 1.e-2*(max_alpha - min_alpha):
        step = 0.5*(alpha_low + alpha_hi)

    return step

################################################################################
#                           Base Line-Search Object                            #
################################################################################

class LineSearch(object):
    """
    Base class for all line search algorithms. Provides rudimentary
    error-checking functionality, and an interface that should be adhered to
    when writing new line search functions.

    Attributes
    ----------
    decr_cond : float
        Sufficient decrease condition.
    max_iter : int
        Maximum iterations for the line search.
    out_file : file
        File stream for writing data.

    Parameters
    ----------
    optns : dict
    out_file : file
    """

    def __init__(self, optns={}, out_file=sys.stdout):
        self.max_iter = get_opt(optns, 50, 'max_iter')
        self.decr_cond = get_opt(optns, 1e-4, 'decr_cond')
        self.out_file = out_file

    def _validate_options(self):
        if not (self.max_iter >= 1):
            raise ValueError('max_iter must be a positive integer')

        if not (0 < self.decr_cond < 1):
            raise ValueError('suff_cond must be 0 < suff_cond < 1')

    def find_step_length(self, merit):
        """
        Find an appropriate step size for the given merit function that leads
        to the minimum in the search direction.

        Parameters
        ----------
        merit : MeritFunc-like
            Merit function object derived from the base MeritFunc class.

        Returns
        -------
        float : Step size.
        int : Number of iterations taken for the search.
        """
        raise NotImplementedError # pragma: no cover

################################################################################
#                          Back Tracking Line-Search                           #
################################################################################

class BackTracking(LineSearch):
    """
    Back-tracking line search.

    Attributes
    ----------
    alpha_init : float
        Initial step size.
    alpha_min : float
        Minimum step size.
    rdtn_factor : float
        Reduction factor for the step size at each iteration.
    p_dot_dfdx : float
        Value of :math:`\\langle p, \\nabla f \\rangle` at current step.
    """
    def __init__(self, optns={}, out_file=sys.stdout):
        super(BackTracking, self).__init__(optns, out_file)
        self.alpha_init = get_opt(optns, 1.0, 'alpha_init')
        self.alpha_max = self.alpha_init
        self.alpha_min = get_opt(optns, 1e-4, 'alpha_min')
        self.rdtn_factor = get_opt(optns, 0.5, 'rdtn_factor')
        self.p_dot_dfdx = 0.0

    def _validate_options(self):
        if not (0 < self.alpha_init <= 1):
            raise ValueError('alpha_init must be 0 < alpha_init <=1')

        if not (0 <= self.rdtn_factor <= 1):
            raise ValueError('reduction factor must be 0 <= rdtn_factor <= 1')

        super(BackTracking, self)._validate_options()

    def find_step_length(self, merit):
        self._validate_options()

        if not isinstance(merit, MeritFunction):
            raise ValueError('unknown merit_function type')

        self.p_dot_dfdx = merit.p_dot_grad

        if (self.p_dot_dfdx >= 0):
            raise ValueError('search direction is not a descent direction')

        alpha = self.alpha_init
        self.f_init = merit.eval_func(alpha)

        self.out_file.write('\n')

        n_iter = 0
        while (alpha > self.alpha_min) and (n_iter < self.max_iter):

            self.out_file.write('   Backtracking Linesearch : iter %i\n'%(n_iter + 1))
            self.out_file.write('   ---------------------------------\n')
            
            f_sufficient = self.f_init + self.decr_cond*alpha*self.p_dot_dfdx
            self.f = merit.eval_func(alpha)
            
            self.out_file.write('   merit val  = %e\n'%self.f)
            self.out_file.write('   sufficient = %e\n'%f_sufficient)
            
            if self.f <= f_sufficient:
                self.out_file.write('\nStep found!\n')
                return alpha, n_iter
            else:
                alpha *= self.rdtn_factor
            n_iter += 1
            
            self.out_file.write('\n')

        # if we got here, linesearch failed
        self.out_file.write('\nLinesearch failed!\n')
        return self.alpha_min, self.max_iter

################################################################################
#                           Strong Wolfe Line-Search                           #
################################################################################

class StrongWolfe(LineSearch):
    """
    Strong Wolfe line search.

    Attributes
    ----------
    alpha_init : float
        Initial step size.
    alpha_max : float
        Maximum step size.
    curv_cond : float
        Curvature condition to be satisfied.
    """
    def __init__(self, optns={}, out_file=sys.stdout):
        super(StrongWolfe, self).__init__(optns, out_file)
        self.alpha_init = get_opt(optns, 1.0, 'alpha_init')
        self.alpha_max = get_opt(optns, 2.0, 'alpha_max')
        self.curv_cond = get_opt(optns, 0.9, 'curv_cond')

    def _validate_options(self):
        if not (self.alpha_init > 0):
            raise ValueError('alpha_init must be greater than zero (0)')

        if (self.alpha_max <= 0.) or (self.alpha_max <= self.alpha_init):
            raise ValueError('alpha_max must be positive and > alpha_init')

        if not (self.decr_cond < self.curv_cond < 1):
            raise ValueError('curv_cond must be suff_cond < curv_cond < 1')

        super(StrongWolfe, self)._validate_options()

    def _zoom(self, alpha_low, alpha_hi, phi_low, phi_hi, dphi_low, dphi_hi):
        merit = self.merit_function

        # if the lower and upper bounds are equal, this means we just hit
        # alpha_max, so we will return alpha_max
        if (alpha_low == alpha_hi) and (alpha_low == self.alpha_max):
            return self.alpha_max, 0
        
        self.out_file.write('\n')

        # START OF BIG FOR-LOOP
        for i in xrange(self.max_iter):
            
            self.out_file.write('    Strong-Wolfe Zoom : iter %i\n'%(i+1))

            # user interpolation to get the new step
            alpha_new = quadratic_step(alpha_low, alpha_hi, phi_low, phi_hi, dphi_low)
            # evaluate the merit function at the interpolated step
            phi_new = merit.eval_func(alpha_new)

            # check if this step violates sufficient decrease
            phi_sufficient = self.phi_init + \
                self.decr_cond*alpha_new*self.dphi_init

            self.out_file.write('    phi low        = %e\n'%phi_low)
            self.out_file.write('    phi hi         = %e\n'%phi_hi)
            self.out_file.write('    phi new        = %e\n'%phi_new)
            self.out_file.write('    phi sufficient = %e\n'%phi_sufficient)
            
            if (phi_new > phi_sufficient) and (phi_new >= phi_low):
                alpha_hi = alpha_new
                phi_hi = phi_new
                self.deriv_hi = False # we don't know the derivative at alpha_hi
            else:
                # now we evaluate merit grad and check curvature condition
                dphi_new = merit.eval_grad(alpha_new)
                if abs(dphi_new) <= -self.curv_cond*self.dphi_init:
                    # we also satisfied curvature condition, so we can stop
                    self.out_file.write('\nStep found!\n')
                    return alpha_new, i
                elif dphi_new*(alpha_hi - alpha_low) >= 0:
                    # curvature condition was not satisfied
                    # alpha_new and alpha_low bracket a minimum
                    alpha_hi = alpha_low
                    phi_hi = phi_low
                    self.deriv_hi = True

                # we satisfied sufficient decrease so the new step is low step
                alpha_low = alpha_new
                phi_low = phi_new
                dphi_low = dphi_new
            
            self.out_file.write('\n')

        # END OF BIG FOR-LOOP
        phi_sufficient = self.phi_init + self.decr_cond*alpha_new*self.dphi_init
        if phi_new < phi_sufficient:
            self.out_file.write(
                '\n>> WARNING : Step found but curvature condition not met! <<\n')
            return alpha_new, i

        # if we got here then we didn't find a step
        raise Exception('StrongWolfe._zoom(): Failed to find a step length!')

    def find_step_length(self, merit):
        self._validate_options()

        if not isinstance(merit, MeritFunction):
            raise ValueError('unknown merit_function type')

        self.merit_function = merit
        self.phi_init = merit.eval_func(0.0)
        self.dphi_init = merit.eval_grad(0.0)

        alpha_old = 0.0
        phi_old = self.phi_init
        dphi_old = self.dphi_init

        if dphi_old > 0.0:
            raise ValueError('search direction is not a descent direction')

        quad_coeff = 0.0
        self.deriv_hi = False

        self.out_file.write('\n')

        # START OF BIG FOR-LOOP
        for i in xrange(self.max_iter):

            self.out_file.write('  Strong-Wolfe Line Search : iter %i\n'%(i+1))
            self.out_file.write('  ----------------------------------\n')

            # get new step
            if i == 0:
                # set alpha to alpha_init if we're at the first step
                alpha_new = self.alpha_init
            else:
                # if it's not the first step, check quad_coeff
                if quad_coeff > 0:
                    # if quad_coeff is positive, then step alpha forward
                    alpha_new = alpha_old - 0.5*dphi_old/quad_coeff
                    if (alpha_new < alpha_old) or (alpha_new > self.alpha_max):
                        alpha_new = min(2*alpha_old, self.alpha_max)
                else:
                    # if quad_coeff is negative, take some minimum between
                    alpha_new = max(2*alpha_old, self.alpha_max)

            # get new function value
            phi_new = merit.eval_func(alpha_new)

            # if new step violates sufficient decrease, call zoom
            phi_sufficient = self.phi_init + \
                self.decr_cond*alpha_new*self.dphi_init

            self.out_file.write('  phi old        = %e\n'%(phi_old))
            self.out_file.write('  phi new        = %e\n'%(phi_new))
            self.out_file.write('  phi sufficient = %e\n'%(phi_sufficient))
            
            if (phi_new > phi_sufficient) or ((i > 0) and (phi_new >= phi_old)):
                dphi_new = 0.0
                self.deriv_hi = False
                out = self._zoom(
                    alpha_old, alpha_new, phi_old, phi_new, dphi_old, dphi_new)
                return out[0], i+out[1]

            # otherwise, get new gradient
            dphi_new = merit.eval_grad(alpha_new)

            # check curvature condition
            if abs(dphi_new) <= -self.curv_cond*self.dphi_init:

                # if curvature condition is satisfied, return alpha_old
                if self.curv_cond > 1.e-6:
                    self.out_file.write('\nStep found!\n')
                    return alpha_new, i

                # a very small curvature is supicious, check for local minimum
                # perturb in one direction first
                perturb = merit.eval_func(alpha_new - self.alpha_max*1.e-6)

                # if the perturbation yielded a smaller function, update
                if perturb < phi_new:
                    phi_new = perturb
                    phi_new = merit.eval_func(
                        alpha_new - self.alpha_max*1.e-6)
                else:
                    # otherwise perturb in the other direction
                    perturb = merit.eval_func(
                        alpha_new + self.alpha_max*1.e-6)
                    # if the perturbation yielded a smaller function, update
                    if perturb < phi_new:
                        phi_new = perturb
                        phi_new = merit.eval_func(
                            alpha_new + self.alpha_max*1.e-6)
                    else:
                        # if neither perturbation worked, we have a true minimum
                        self.out_file.write('\nStep found!\n')
                        return alpha_new, i

            # check if new gradient is positive
            if dphi_new >= 0:
                # if we get this far, the curvature condition is not satisfied
                self.deriv_hi = True
                out = self._zoom(
                    alpha_old, alpha_new, phi_old, phi_new, dphi_old, dphi_new)
                return out[0], i+out[1]

            # update old variables
            quad_coeff = alpha_new - alpha_old
            quad_coeff = ((phi_new - phi_old) -
                          dphi_new*quad_coeff)/(quad_coeff**2)
            alpha_old = alpha_new
            phi_old = phi_new
            dphi_old = dphi_new

            self.out_file.write('\n')

        # END OF BIG FOR-LOOP

        # if we got here then we didn't find a step
        raise Exception('StrongWolfe.find_step_lenght(): ' +
                        'failed to find a step length')
