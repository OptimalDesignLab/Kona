
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



