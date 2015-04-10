
"""
# question:
# 3  line 85 dsyev_
# 4  line 140 doubts
# 5  line 153 228 c++ iterator counterparts in python??
# 7  line 512  recast in c++ ??                                   #



"""

import numpy as np

"""
Translated from krylov.cpp

Version on the move...
"""

# const double kEpsilon = numeric_limits<double>::epsilon();   Tranlate this!!

kEpsilon = np.finfo(float).eps

#===================================================================================================

def sign(x, y):
    """x = sign(y)* abs(x)"""

    if y == 0.0:
        return 0.0
    else:
        if y < 0:
            return -abs(x)
        else:
            return abs(x)


def CalcEpsilon(eval_at_norm, mult_by_norm):
    """ Type:
    eval_at_norm : double
    mult_by_norm : double   """

    if (mult_by_norm < kEpsilon*eval_at_norm) or (mult_by_norm < kEpsilon):
        # multiplying vector is zero or essentially zero
        return 1.0
    else:
        # multiplying vector dominates, so treat eval_at vector like zero
        if (eval_at_norm < kEpsilon*mult_by_norm):
            return np.sqrt(kEpsilon)/mult_by_norm
        else:
            return np.sqrt(kEpsilon)*eval_at_norm/mult_by_norm


def eigenvalues(A):
    eig_vals, eig_vec = np.linalg.eig(A)
    idx = eig_vals.argsort() # eig doesn't sort, so we have to
    eig_vals = eig_vals[idx]
    eig_vec = eig_vec[idx]

    return eig_vals, eig_vec

def applyGivens(s, c, h1, h2):

    # clockwise rotation? 

    temp = c*h1 + s*h2
    h2 = c*h2 - s*h1
    h1 = temp

    return h1,h2



def generateGivens(dx, dy, s, c):

    if (dx == 0.0) and (dy == 0.0):
        c = 1.0
        s = 0.0
    elif abs(dy) > abs(dx):
        tmp = dx/dy
        dx = np.sqrt(1.0 + tmp*tmp)
        s = sign(1.0/dx, dy)
        c = tmp*s
    elif abs(dy) <= abs(dx):
        tmp = dy/dx
        dy = np.sqrt(1.0 + tmp*tmp)
        c = sign(1.0/dy, dx)
        s = tmp*c

    else:   # dx and/or dy must be invalid
        dx = 0.0
        dy = 0.0
        c = 1.0
        s = 0.0

    dx = abs(dx*dy)
    dy = 0.0
    return dx, dy
    
def solve_trust_reduced(n, H, g, radius):
    """
    Solves the reduced-space trust-region subproblem

    Parameters
    ----------
    n : int
        size of the reduced space
    H : numpy 2d array
        reduced-space Hessian
    g : numpy array
        gradient in the reduced space
    radius : float
        trust-region radius

    Returns
    -------
    y : numpy array
        solution to reduced-space trust-region problem
    lam : float
        Lagrange multiplier value
    pred : float
        predicted decrease in the objective
    """
    if n < 0:
        raise ValueError('reduced-space dimension must be greater than 0')
    if radius < 0.0:
        raise ValueError('trust-region radius must be nonnegative: radius = ' + radius)
    if H.shape[0] != n or H.shape[1] != n:
        raise Exception('reduced Hessian shape inconsistent with n')
    if g.shape[0] != n:
        raise Exception('reduced gradient shape inconsistent with n')

    eig_vals, eig = eigenvalues(H)
    eigmin = eig[0]
    lam = 0.0
    if eigmin > 1e-12:
        # Hessian is semi-definite on span(Z), so solve for y and check if ||y||
        # is in trust region radius
        y, fnc, dfnc = trust_function(n, H, g, lam, radius)
        if (fnc < 0.0): # i.e. norm_2(y) < raidus
            # compute predicted decrease in objective and return
            pred = -y.dot(0.5*H.dot(y) + g)
            return y, lam, pred

    # if we get here, either the Hessian is semi-definite or ||y|| > radius

    # find upper bound for bracket, lam < lam_h
    max_brk = 20
    dlam = 0.1*max(-eigmin, kEpsilon)
    lam_h = max(-eigmin, 0.0) + dlam
    y, fnc_h, dfnc = trust_function(n, H, g, lam_h, radius)
    for k in range(max_brk):
        if fnc_h > 0.0:
            break
        dlam *= 0.1
        lam_h = max(-eigmin, 0.0) + dlam
        y, fnc_h, dfnc = trust_function(n, H, g, lam_h, radius)

    # find lower bound for bracket, lam_l < lam
    dlam = sqrt(kEpsilon)
    lam_l = max(-eigmin, 0.0) + dlam
    y, fnc_l, dfnc = trust_function(n, H, g, lam_l, radius)
    for k in range(max_brk):
        if fnc_l < 0.0:
            break
        dlam *= 100.0
        lam_l = max(-eigmin, 0.0) + dlam
        y, fnc_l, dfnc = trust_function(n, H, g, lam_l, radius)

    # Apply (safe-guarded) Newton's method to find lam
    max_Newt = 50
    lam = 0.5*(lam_l + lam_h)
    dlam_old = abs(lam_h - lam_l)
    dlam = dlam_old
    tol = sqrt(kEpsilon)
    lam_tol = tol*dlam
    y, fnc, dfnc = trust_function(n, H, g, lam, radius)
    res0 = abs(fnc)
    for l in range(max_Newt):
        # check if y lies on the trust region; if so, exit loop
        if abs(fnc) < tol*res0 or abs(dlam) < lam_tol:
            break
        # choose safe-guarded step
        if ((lam - lam_h)*dfnc - fnc)*((lam - lam_l)*dfnc - fnc) > 0.0 or
            abs(2.0*fnc) > abs(dlam_old*dfnc):
            #  use bisection if Newton-step is out of range or not decreasing fast
            dlam_old = dlam
            dlam = 0.5*(lam_h - lam_l);
            lam = lam_l + dlam;
            if (lam_l == lam):
                break
        else:
            # Newton-step is acceptable
            dlam_old = dlam
            dlam = fnc/dfnc
            temp = lam
            lam -= dlam
            if (temp == lam):
                break

        # evaluate function at new lambda
        y, fnc, dfnc = trust_function(n, H, g, lam, radius)
        if fnc < 0.0:
            lam_l = lam
        else:
            lam_h = lam

    if l == max_Newt: # Newton's method failed to converge
        raise Exception('Newton\'s method failed to converge to a valid lambda')

    # compute predicted decrease in objective
    pred = -y.dot(0.5*H.dot(y) + g)
    return y, lam, pred
