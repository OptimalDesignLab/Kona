
"""
# question:
# 3  line 85 dsyev_
# 4  line 140 doubts
# 5  line 153 228 c++ iterator counterparts in python??
# 7  line 512  recast in c++ ??                                   #



"""


# import numpy
from numpy import sqrt
import numpy as np
from usertemplate import UserTemplate
from usermemory import UserMemory
from vectors import DesignVector

"""
Translated from krylov.cpp

Version on the move...
"""

# const double kEpsilon = numeric_limits<double>::epsilon();   Tranlate this!!

kEpsilon = np.finfo(float).eps

def sign(x, y):

    "x: float     y: float"

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
            return sqrt(kEpsilon)/mult_by_norm
        else:
            return sqrt(kEpsilon)*eval_at_norm/mult_by_norm

def eigenvalues(A, eig):
    egi_vals, eig_vec = np.linalg.eig(A)
    idx = eig_vals.argsort() # eig doesn't sort, so we have to
    eig_vals = eig_vals[idx]
    eig_vec = eig_vec[idx]

    return eig_vals, eig_vec


#===================================================================================================


# def factorQR(nrow, ncol, A, QR):

#     # if (nrow < 1) or (ncol < 1):
#     #     raise ValueError('krylov.cpp (factorQR): matrix dimensions must be greater than 0.')

#     # if (A.shape[0] < nrow) or (A.shape[1] < ncol):
#     #     raise ValueError('krylov.cpp (factorQR): given matrix has fewer rows/columns than given dimensions.')

#     # if nrow < ncol:
#     #     raise ValueError('krylov.cpp (factorQR): number of rows must be greater than or equal the number of columns.')

#     # # Copy A into QR in column-major ordering
#     # QR.reshape(nrow*ncol + ncol)
#     # for j in range(ncol):
#     #     for i in range(nrow):
#     #         QR[j*nrow + i] = A[i,j]

#     # m = nrow
#     # n = ncol
#     # lwork = ncol

#     # work = np.zeros(lwork)

#     # ##########################
#     # # ublas::vector<double>::iterator tau = QR.end() - ncol;    ## ???
#     # # dgeqrf_(&m, &n, &*QR.begin(), &m, &*tau, &*work.begin(), &lwork, &info);

#     q, r = np.linalg.qr(QR)
#     ##########################

#     if info!=0:
#         raise ValueError('krylov.cpp (factorQR): LAPACK routine dgeqrf failed with info ='  + info)


#===================================================================================================


# def solveR(nrow, ncol, QR, b, x, transpose):

#     if (nrow < 1) or (ncol < 1):
#         raise ValueError('krylov.cpp (invertR): matrix dimensions must be greater than 0.')

#     if nrow < ncol:
#         raise ValueError('krylov.cpp (invertR): number of rows must be greater than or equal the number of columns.')

#     if (len(b) < ncol) or (len(x) < ncol):
#         raise ValueError('krylov.cpp (invertR): vector b and/or x are smaller than size of R.')

#     # copy rhs vector b into x (which is later overwritten with solution)
#     x = b

#     uplo = 'U'
#     trans = 'N'

#     if transpose:
#         trans = 'T'

#     diag = 'N'
#     m = nrow
#     n = ncol
#     nrhs = 1

#      ####################
#     dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, &*QR.begin(), &m, &*x.begin(), &n, &info);
#     ####################

#     if info>0 :
#         raise ValueError('krylov.cpp (solveR): the' + info + '-th diagonal entry of R is zero')

#     else:
#         if info < 0:
#             raise ValueError('krylov.cpp (solveR): LAPACK routine dtrtrs failed with info =' + info)


#===================================================================================================


# def applyQ(nrow, ncol, QR, b, x, transpose):

#     if (nrow < 1) or (ncol < 1):
#         raise ValueError('krylov.cpp (applyQ): matrix dimensions must be greater than 0.')

#     if nrow < ncol:
#         raise ValueError('krylov.cpp (applyQ): number of rows must be greater than or equal the number of columns.')

#     if (len(b) < nrow) or (len(x) < nrow):
#         raise ValueError('krylov.cpp (applyQ): vector b and/or x are smaller than nrow.')

#     # copy rhs vector b into x (which is later overwritten with solution)
#     x = b
#     side = 'L'
#     trans = 'N'

#     if transpose:
#         trans = 'T'

#     m = nrow
#     n = ncol
#     nrhs = 1
#     lwork = nrow

#     work = np.zeros(lwork)

#     ##########################
#     ublas::vector<double>::iterator tau = QR.end() - ncol;
#     dormqr_(&side, &trans, &m, &nrhs, &n, &*QR.begin(), &m, &*tau, &*x.begin(),&m, &*work.begin(), &lwork, &info);
#     ##########################

#     if info!=0:
#         raise ValueError('krylov.cpp (applyQ): LAPACK routine dormqr failed with info = ' + info)


#===================================================================================================


# def factorCholesky(n, A, UTU):

#     if n<1:
#         raise ValueError('krylov.cpp (factorCholesky): matrix dimension must be greater than 0.')

#     if (A.shape[0] < n) or (A.shape[1] < n):
#         raise ValueError('krylov.cpp (factorCholesky): given matrix has fewer rows/columns than given dimension.')

#     # UTU stores the symmetric part of A in column-major ordering (actually,
#     # ordering doesn't matter for symmetric matrix)
#     UTU.reshape(n*n)

#     for i in range(n):
#         for j in range(i,n):
#             UTU[i*n + j] = 0.5*(A[i,j] + A[j,i])
#             UTU[j*n + i] = UTU[i*n + j]

#     uplo = 'U'
#     Adim = n

#     ###########################
#     dpotrf_(&uplo, &Adim, &*UTU.begin(), &Adim, &info);
#     ###########################

#     if info < 0:
#         raise ValueError('krylov.cpp (factorCholesky): LAPACK routine dpotrf failed with info =' + info)


#===================================================================================================

# def solveU(n, UTU, b, x, transpose):

#     if n<1:
#         raise ValueError('krylov.cpp (applyU): matrix dimension must be greater than 0.')

#     if (b.size() < n) or (x.size() < n):
#         raise ValueError('krylov.cpp (applyU): vector b and/or x are smaller than n.')

#     x = b

#     side = 'L'
#     uplo = 'U'
#     trans = 'N'

#     if transpose:
#         trans = 'T'

#     diag = 'N'
#     nrow = n
#     nrhs = 1
#     one = 1.0

#     dtrsm_(&side, &uplo, &trans, &diag, &nrow, &nrhs, &one, &*UTU.begin(),
#         &nrow, &*x.begin(), &nrow)


#===================================================================================================


# def computeSVD(nrow, ncol, A, Sigma, U, VT, All_of_U):

#     if (nrow < 1) or (ncol < 1):
#         raise ValueError('krylov.cpp (computeSVD): matrix dimensions must be greater than 0.')

#     if (A.size1() < nrow) or (A.size2() < ncol):
#         raise ValueError('krylov.cpp (computeSVD): given matrix has fewer rows/columns than given dimensions.')

#     if nrow < ncol:
#         raise ValueError('krylov.cpp (computeSVD): number of rows must be greater than or equal the number of columns.')

#      if All_of_U:
#         # Resize output vectors and copy A into work array
#         Sigma.reshape(ncol)
#         U.reshape(nrow*nrow)
#         VT.reshape(ncol*ncol)
#         Acpy = np.zeros(nrow*ncol)
#         for j in range(ncol):
#             for i in range(nrow):
#                 Acpy[j*nrow + i] = A[i,j]

#         jobu = 'A'
#         jobvt = 'A'
#         m = nrow
#         n = ncol
#         lwork = max(3*ncol + nrow, 5*ncol)
#         work = np.zeros(lwork)

#         ############################
#         dgesvd_(&jobu, &jobvt, &m, &n, &*Acpy.begin(), &m, &*Sigma.begin(),
#                 &*U.begin(), &m, &*VT.begin(), &n, &*work.begin(), &lwork, &info);
#         ############################

#     else
#         # Resize output vectors, and copy A into U in column-major ordering
#         Sigma.reshape(ncol)
#         U.reshape(nrow*ncol)
#         VT.reshape(ncol*ncol)
#         for j in range(ncol):
#             for i in range(nrow):
#                 U[j*nrow + i] = A[i,j]

#         jobu = 'O'
#         jobvt = 'A'
#         m = nrow
#         n = ncol

#         ##########################
#         double * Ujunk;
#         ##########################
#         ldu = 1

#         lwork = max(3*ncol + nrow, 5*ncol)
#         work = np.zeros(lwork)

#         ##############################
#         dgesvd_(&jobu, &jobvt, &m, &n, &*U.begin(), &m, &*Sigma.begin(), Ujunk, &ldu,
#                 &*VT.begin(), &n, &*work.begin(), &lwork, &info)
#         ##############################

#         if (info != 0):
#             raise ValueError('krylov.cpp (computeSVD): LAPACK routine dgesvd failed with info =' + info)


#=========================================================================

def applyGivens(s, c, h1, h2):

    temp = c*h1 + s*h2
    h2 = c*h2 - s*h1
    h1 = temp


#=========================================================================

def generateGivens(dx, dy, s, c):

    if (dx == 0.0) and (dy == 0.0):
        c = 1.0
        s = 0.0
    elif abs(dy) > abs(dx):
        tmp = dx/dy
        dx = sqrt(1.0 + tmp*tmp)
        s = sign(1.0/dx, dy)
        c = tmp*s
    elif fabs(dy) <= fabs(dx):
        tmp = dy/dx
        dy = sqrt(1.0 + tmp*tmp)
        c = sign(1.0/dy, dx)
        s = tmp*c

    else:   # dx and/or dy must be invalid
        dx = 0.0
        dy = 0.0
        c = 1.0
        s = 0.0

    dx = abs(dx*dy)
    dy = 0.0


#=========================================================================

# def solveReduced(n, A, rhs, x):

#     if n<1:
#         raise ValueError('krylov.cpp (solveReduced): matrix dimensions must be greater than 0.')

#     if (A.shape[0] < n) or (A.shape[1] < n):
#         raise ValueError('krylov.cpp (solveReduced): given matrix has fewer rows/columns than given dimensions.')

#     if len(rhs) < n:
#         raise ValueError('krylov.cpp (solveReduced): given rhs has fewer rows than given dimensions.')

#     if len(x) < n:
#         raise ValueError('krylov.cpp (solveReduced): given x has fewer rows than given dimensions.')

#     # LU stores A in column-major ordering (eventually, LU will hold the
#     # LU-factorization of A

#     LU = np.zeros(n*n)
#     for j in range(n):
#         for i in range(n):
#             LU[j*n + i] = A[i,j]

#     # Y stores RHS in column-major ordering (Y is overwritten with solution)
#     Y = np.zeros(n)
#     for i in range(n):
#         Y[i] = rhs[i]

#     Arow = n      # need copy, because n is const int &
#     RHScol = 1    # similarly

#     ipiv = np.zeros(n)

#     #########################
#     dgesv_(&Arow, &RHScol, &*LU.begin(), &Arow, &*ipiv.begin(), &*Y.begin(), &Arow,
#     &info)
#     #########################

#     if info!=0:
#         raise ValueError('krylov.cpp (solveReduced): LAPACK routine dgesv failed with info = ' + info)

#     # put solution into x
#     for i in range(n):
#         x[i] = Y[i]


#=========================================================================


# def solveReducedMultipleRHS(n, A, nrhs, RHS, X):

#     if (n < 1) or (nrhs < 1):
#         raise ValueError('krylov.cpp (solveReducedMultipleRHS): matrix dimensions must be greater than 0.')

#     if (A.shape[0] < n) or (A.shape[1] < n):
#         raise ValueError('krylov.cpp (solveReducedMultipleRHS): given matrix has fewer rows/columns than given dimensions.')

#     if (RHS.shape[0] < n) or (RHS.shape[1] < nrhs):
#         raise ValueError('krylov.cpp (solveReducedMultipleRHS): given RHS has fewer rows/columns than given dimensions.')

#     if (X.shpae[0] < n) or (X.shape[1] < nrhs):
#         raise ValueError('krylov.cpp (solveReducedMultipleRHS): given X has fewer rows/columns than given dimensions.')

#     # LU stores A in column-major ordering (eventually, LU will hold the
#     # LU-factorization of A
#     LU = np.zeros(n*n)
#     for j in range(n):
#         for i in range(n):
#             LU[j*n + i] = A[i,j]


#     # Y stores RHS in column-major ordering (Y is overwritten with solution)
#     Y = np.zeros(n*nrhs)
#     for j in range(nrhs):
#         for i in range(n):
#             Y[j*n + i] = RHS[i,j]


#     Arow = n        #  need copy, because n is const int &
#     RHScol = nrhs   # similarly

#     rcond = 1.e-12

#     S = np.zeros(n)
#     iwork = np.zeros(20*n)

#     lwork = -1;
#     work = np.zeros(1)

#     ###############################
#     dgelsd_(&Arow, &Arow, &RHScol, &*LU.begin(), &Arow, &*Y.begin(), &Arow,
#     &*S.begin(), &rcond, &rank, &*work.begin(), &lwork, &*iwork.begin(),
#     &info)

#     lwork = static_cast<int>(work(0))
#     ###############################

#     work.reshape(lwork)

#     ###############################
#     dgelsd_(&Arow, &Arow, &RHScol, &*LU.begin(), &Arow, &*Y.begin(), &Arow,
#     &*S.begin(), &rcond, &rank, &*work.begin(), &lwork, &*iwork.begin(),
#     &info)
#     ###############################

#     if info!=0:
#         raise ValueError('krylov.cpp (solveReducedMultipleRHS): LAPACK routine dgelsd failed with info = ' + info)

#     for j in range(nrhs):
#         for i in range(n):
#             X[i,j] = Y[j*n + i]


# def solveReducedHessenberg(n, Hsbg, rhs, x):

#     # initialize...
#     x = rhs
#     # ... and backsolve
#     for i in range(n-1, -1, -1):
#         x[i] = x[i]/Hsbg[i,i]
#         for j in range(i-1, -1, -1):
#             x[j] = x[j] - Hsbg[j,i]*x[i]


#=======================================================

def trustFunction(H, g, lambda2, radius):
    # First, factorize the matrix [H + lambda*I], where H is the reduced Hessain
    diag = max(1.0, lambda2)*0.01*kEpsilon
    semidefinite = true
    regiter = 0

    Hhat = np.zeros(n,n)
    work = np.zeros(n)

    while semidefinite:
        regiter = regiter + 1

        try:
            for i in range(n):
                for j in range(n):
                    Hhat[i,j] = H[i,j]
                    Hhat[i,i] = Hhat[i,i] + lambda2 + diag

            factorCholesky(n, Hhat, UTU)
            semidefinite = false

        except factor_failed:
            diag = diag*100.0

        #ifdef VERBOSE_DEBUG
          raise ValueError('\t\ttrustFunction: factorCholesky() failed,'
           + " adding " + diag + " to diagonal..." )
        #endif

        if (regiter > 20):
            raise ValueError('krylov.cpp (trustFunction): regularization of Cholesky factorization failed.')

    # Next, solve for the step; the step's length is used to define the objective
    y = np.zeros(n)
    solveU(n, UTU, g, work, true)
    solveU(n, UTU, work, y, false)
    y = y*(-1.0)     # to "move" g to rhs

    # compute the function
    norm_y = norm_2(y)
    fnc = 1.0/radius - 1.0/norm_y

    # find derivative of the function
    solveU(n, UTU, y, work, true)
    norm_work = norm_2(work)
    dfnc = norm_work/norm_y
    dfnc = -(dfnc*dfnc)/norm_y

    # return step, function, and derivative
    return boost::make_tuple(y, fnc, dfnc)


def solveTrustReduced(n, H, g, radius):
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

    eig_vals, eig = eigenvalues(H)
    eigmin = eig[0]
    lam = 0.0
    if eigmin > 1e-12:
        # Hessian is semi-definite on span(Z), so solve for y and check if ||y||
        # is in trust region radius
        y_tmp, fnc, dfnc = trust_function(n, H, g, lam, radius)
        if (fnc < 0.0): # i.e. norm_2(y) < raidus
            # // compute predicted decrease in objective
            pred = 0.0
            y = y_tmp[0:n].copy()
            pred = -y[0:n].dot(0.5*H[0:n][0:n].dot(y[0:n]) + g[0:n])
            
            for i in range(n):
                for j in range(n):
                    pred = pred - 0.5*y[i]*H[i,j]*y[j]
                    pred = pred - g[i]*y[i]

            # pred = rhs(0)*y(0) - 0.5*pred;
            return

        # #ifdef DEBUG
        #     cout << "\t\tsolveTrustReduced: norm_2(y) = " << norm_2(y_tmp) << " > "
        #          << radius << endl;
        # #endif

    # // if we get here, either the Hessian is semi-definite or ||y|| > radius
    # // bracket the Lagrange multiplier lambda

    max_brk = 20
    dlam = 0.1*max(-eigmin, kEpsilon)
    lambda_h = max(-eigmin, 0.0) + dlam

    ######################################
    boost::tie(y_tmp, fnc_h, dfnc) = trustFunction(n, H, g, lambda_h, radius)
    ######################################

    for k in range(max_brk):

        # #ifdef VERBOSE_DEBUG
        # cout << "\t\tsolveTrustReduced: (lambda_h, fnc_h) = ("
        # << lambda_h << ", " << fnc_h << ")" << endl;
        # #endif

        if fnc_h > 0.0:
            break

        dlam = dlam*0.1

        lambda_h = max(-eigmin, 0.0) + dlam

        #######################
        boost::tie(y_tmp, fnc_h, dfnc) = trustFunction(n, H, g, lambda_h, radius);
        #######################

    dlam = sqrt(kEpsilon)
    lambda_l = max(-eigmin, 0.0) + dlam

    ##########################3
    boost::tie(y_tmp, fnc_l, dfnc) = trustFunction(n, H, g, lambda_l, radius);
    ##########################

    for (int k = 0; k < max_brk; k++) {
    #ifdef VERBOSE_DEBUG
    cout << "\t\tsolveTrustReduced: (lambda_l, fnc_l) = ("
    << lambda_l << ", " << fnc_l << ")" << endl;
    #endif
    if (fnc_l < 0.0) break;
    dlam *= 100.0;
    lambda_l = std::max(-eigmin, 0.0) + dlam;
    boost::tie(y_tmp, fnc_l, dfnc) = trustFunction(n, H, g, lambda_l, radius);
    }


    lambda2 = 0.5*(lambda_l + lambda_h)


    ############################
    # #ifdef VERBOSE_DEBUG
    #     cout << "\t\tsolveTrustReduced: initial lambda = " << lambda << endl;
    # #endif
    ###########################

    # Apply (safe-guarded) Newton's method to find lambda2
    dlam_old = fabs(lambda_h - lambda_l)
    dlam = dlam_old

    maxNewt = 50
    tol = sqrt(kEpsilon)
    lam_tol = sqrt(kEpsilon)*dlam

    ###################################
    boost::tie(y_tmp, fnc, dfnc) = trustFunction(n, H, g, lambda, radius)
    ###################################
    res0 = abs(fnc)

    for l in range(maxNewt):
        # check if y lies on the trust region; if so, exit

        ######################
        #ifdef VERBOSE_DEBUG
            # cout << "\t\tsolveTrustReduced: Newton iter = " << l << ": res = "
            #     << fabs(fnc) << ": lambda = " << lambda << endl;
            # #endif

        # if  (fabs(fnc) < tol*res0) or fabs(dlam) < lam_tol)
        #     # #ifdef DEBUG
        #         cout << "\t\tsolveTrustReduced: Newton converged with lambda2 = "
        #         << lambda2 << endl
        #     # #endif
        #     break

        # choose safe-guarded step
        if ( ( ((lambda2 - lambda_h)*dfnc - fnc)*((lambda2 - lambda_l)*dfnc - fnc) > 0.0) or
            (abs(2.0*fnc) > abs(dlam_old*dfnc) ) ) {
            #  use bisection if Newton-step is out of range or not decreasing fast
            dlam_old = dlam
            dlam = 0.5*(lambda_h - lambda_l);

            lambda2 = lambda_l + dlam;
            if (lambda_l == lambda2):
                break
        else:
            # Newton-step is acceptable
            dlam_old = dlam
            dlam = fnc/dfnc
            temp = lambda2
            lambda2 = lambda2 - dlam
            if (temp == lambda2):
                break

        # evaluate function at new lambda
        ############################
        boost::tie(y_tmp, fnc, dfnc) = trustFunction(n, H, g, lambda, radius)
        #############################
        if fnc < 0.0:
            lambda_l = lambda2
        else:
            lambda_h = lambda2

    if (l == maxNewt):
    # Newton's method failed to converge
        raise ValueError('krylov.cpp (solveTrustReduced): Newtons method failed to converge to a valid lambda')

    # compute predicted decrease in objective
    pred = 0.0
    for i in range(n):
        y[i] = y_tmp[i]

    for i in range(n):
        for j in range(n):
            pred = pred - 0.5*y[i]*H[i,j]*y[j]
            pred = pred - g[i]*y[i]


# =================================================================




# =================================================================

def solveUnderdeterminedMinNorm(nrow, ncol, A, b, x):

    if ( nrow < 0 ) or (ncol < 0):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm): matrix dimensions must be greater than 0.')

    if (nrow > ncol):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm): expecting rectangular matrix with nrow <= ncol.')

    if (A.shape[0] < nrow) or (A.shape[1] < ncol):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm):  A matrix sizes inconsistent with nrow and ncol.')

    if (len(b) < nrow) or (len(x) < ncol):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm): b and/or x vectors are smaller than necessary.')


    #  computeSVD(ncol, nrow, A.transpose, Sigma, P, QT, true)

    PTmat, Sigma, Qmat = np.linalg.svd(A.T)

    # PTmat = np.empty(ncol, ncol)
    # Qmat = np.empty(nrow, nrow)

    # for k in xrange(ncol):
    #     for j in xrange(ncol):
    #         PTmat[k,j] = P[k*ncol+j]
    # for k in xrange(nrow):
    #     for j in xrange(nrow):
    #         Qmat[k,j] = QT[k*nrow+j]

    solveUnderdeterminedMinNorm(nrow, ncol, Sigma, Qmat, PTmat, b, x)

# =================================================================


def solveUnderdeterminedMinNorm(nrow, ncol, Sigma, U, VT, b,x):

    if ( nrow < 0 ) or (ncol < 0):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm): matrix dimensions must be greater than 0.')

    if (nrow > ncol):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm): expecting rectangular matrix with nrow <= ncol.')

    if ( (U.shape[0] < nrow) or (U.shape[1] < nrow) or \
        (VT.shape[0] < nrow) or (VT.shape[1] < ncol) or \
        (len(Sigma) < nrow)):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm): ' +
            'matrices of left or right singular vectors, or Sigma matrix,' +
            'are smaller than necessary.')

    if (len(b) < nrow) or (len(x) < ncol):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm):'
            'b and/or x vectors are smaller than necessary.')

    # compute rank of A = U*Sigma*VT
    rank = 0
    for i in range(nrow):
        if Sigma[i] > 0.01*Sigma[0]:
            rank = rank + 1     # kEpsilon*Sigma(0)) rank++;

    if (rank == 0) or (Sigma(0) < kEpsilon):
        raise ValueError('krylov.cpp (solveUnderdeterminedMinNorm): singular values are all zero or negative.')

    y = np.zeros(rank)
    for i in range(rank):
        y[i] = 0.0
        for j in range(nrow):
            y[i] = y[i] + U[j,i]*b[j]
            y[i] = y[i]/Sigma[i]

        for j in range(ncol):
            x[j] = 0.0
            for i in range(rank):
                x[j] += y[i]*VT[i,j]


# =================================================================
def solveLeastSquares(nrow, ncol, A, b, x):

    if ( nrow < 0 ) or (ncol < 0):
        raise ValueError('krylov.cpp (solveLeastSquares): matrix dimensions must be greater than 0.')

    if (nrow < ncol):
        raise ValueError('krylov.cpp (solveLeastSquares): expecting rectangular matrix with nrow >= ncol.')

    # copy A into Awrk in column major ordering
    Awrk = np.zeros(nrow*ncol)
    for j in range(ncol):
        for i in range(nrow):
            Awrk[j*nrow + i] = A[i,j]


    trans = 'N'
    m = nrow
    n = ncol
    nrhs = 1
    rhs = b
    lwork = ncol + nrow;
    work = np.zeros(lwork)

    #############################
    dgels_(&trans, &m, &n, &nrhs, &*Awrk.begin(), &m, &*rhs.begin(),
        &m, &*work.begin(), &lwork, &info)
    #############################
    if (info != 0):
        raise ValueError('krylov.cpp (solveLeastSquares): LAPACK routine dgels failed with info = ' + info)

    for i in range(ncol):
        x[i] = rhs[i]

# =================================================================

def solveLeastSquaresOverSphere(nrow, ncol, radius, Sigma, U, VT, b, x):
    if ( nrow < 0 ) or (ncol < 0):
        raise ValueError('krylov.cpp (solveLeastSquaresOverSphere): matrix dimensions must be greater than 0.')

    if (nrow < ncol):
        raise ValueError('krylov.cpp (solveLeastSquaresOverSphere): expecting rectangular matrix with nrow >= ncol.')

    if (radius < 0.0):
        raise ValueError('krylov.cpp (solveLeastSquaresOverSphere): sphere radius must be nonnegative: radius = ' + radius)

    if ( (U.shape[0] < nrow) or (U.shape[1] < ncol) or \
        (VT.shape[0] < ncol) or (VT.shape[1] < ncol) or \
        (len(Sigma) < ncol)):
        raise ValueError('krylov.cpp (solveLeastSquaresOverSphere): ' +
            'matrices of left or right singular vectors, or Sigma matrix,' +
            'are smaller than necessary.')

    if (len(b) < nrow) or (len(x) < ncol):
        raise ValueError('krylov.cpp (solveLeastSquaresOverSphere):'
            'b and/or x vectors are smaller than necessary.')

    # compute rank of A = U*Sigma*VT
    rank = 0
    for i in range(ncol):
        if (Sigma(i) > kEpsilon*Sigma(0)):
            rank = rank + 1

    if (rank == 0) or (Sigma(0) < kEpsilon):
    raise ValueError('krylov.cpp (solveLeastSquaresOverSphere): singular values are all zero or negative.')

    # compute the tentative reduced-space solution, y = Sigma^{-1} U^T b
    g = np.zeros(rank)
    y = np.zeros(rank)

    sol_norm = 0.0;
    for i in range(rank):
        for j in range(nrow):
            g[i] += U[j,i]*b[j]
    y[i] = g[i]/Sigma[i]
    sol_norm = sol_norm + y[i]*y[i]

    rad2 = radius*radius
    lambda2 = 0.0
    if (sol_norm > rad2):
        # tentative solution is outside sphere, so solve secular equation
        phi = sol_norm - rad2
        phi0 = phi
        kMaxIter = 40
        kTol = 100.0*kEpsilon

        for k in range(kMaxIter):
            print ('iter' + k + ": res = " + abs(phi))

            if (abs(phi) < kTol*phi0):
                break

            dphi = 0.0

            for i in range(rank):
                dphi = dphi - 2.0*pow(Sigma[i]*g[i], 2.0)/pow(Sigma[i]]*Sigma[i]] + lambda2, 3.0)

            lambda2 = lambda2 - phi/dphi
            sol_norm = 0.0

            for i in range(rank):
                y[i] = Sigma[i]]*g[i]/(Sigma[i]*Sigma[i] + lambda2)
                sol_norm = sol_norm + y[i]*y[i]

            phi = sol_norm - rad2;


        if k == kMaxIter:
            print 'fabs(phi) = ' + fabs(phi)
            print 'kTol*phi0 = ' + kTol*phi0
            raise ValueError('krylov.cpp (solveLeastSquaresOverSphere): maximum number of Newton iterations exceeded.')

    for j in range(ncol):
        x[j]] = 0.0
        for i in range(rank):
            x[j] = x[j] + y[j]*VT[i,j]


# =================================================================

def double trustResidual(n, H, B, g, y, lambda2):

    if ( n < 0 ):
        raise ValueError('krylov.cpp (trustResidual): matrix dimensions must be greater than 0.')

    if (H.shape[0] < n+1) or (H.shape[1] < n) or (B.shape[0] < n+1) or (B.shape[1] < n):
        raise ValueError('krylov.cpp (trustResidual): H and/or B matrix are smaller than necessary.')

    if (len(g) < 2*n+1) or (len(y) < n):
        raise ValueError('krylov.cpp (trustResidual): g and/or y vectors are smaller than necessary.')


    # // find SVD of B, and find index s such that
    # //     1 = Sigma(0) = --- = Sigma(s-1) > Sigma(s)
    # // Note: for the given B, the Sigma should lie between 0 and 1 (see pg 603,
    # // Golub and van Loan); check this here too


    Sigma = np.array
    P = np.array
    UT = np.array
    computeSVD(n+1, n, B, Sigma, P, UT)

    for s in range(n):
        if (Sigma[s] < - 1E+4*kEpsilon or (Sigma[s] > 1.0+1E+4*kEpsilon):
            raise ValueError('krylov.cpp (trustResidual): Singular values of B are not in [0,1]. check that Z and V have orthogonal columns')
            print "B sigma = "
            for i in range(n):
                print Sigma[i]

    for s in range(n):
        if (abs(Sigma[s] - 1.0) > sqrt(kEpsilon)):
            break                       # //1E+4*kEpsilon) break;

    # #ifdef DEBUG
    # cout << "\t\ttrustResidual: singular values = ";
    # for (int i = 0; i < n; i++)
    # cout << std::setw(16) << std::setprecision(12) << Sigma(i) << " ";
    # cout << endl;
    # cout << "\t\ttrustResidual: n, s, n-s = " << n << ", " << s << ", "
    # << n-s << endl;
    # #endif



    # // compute the matrix C = W^T * Z = N*UT(s:n-1)*(I - B^T*B)
    # // where N^{-1} = diag(sqrt(1 - Sigma(s:n-1)))

    if (s < n):
        C.reshape(n-s,n)
        # // Step 1: compute P := I - B^T*B (stored column major)
        for j in range(n):
            for i in range(n):
                P[i+j*n] = 0.0
                for k in range(n+1):
                    P[i+j*n] = P[i+j*n] - B[k,i]*B[k,j]

            P[j+j*n] += 1.0


        # // Step 2: compute C := QT(s:n-1)*P
        for i in range(n-s):
            for j in range(n):
                C[i,j] = 0.0
                for k in range(n):
                    C[i,j] = C[i,j] + UT[s+i+k*n]*P[k+j*n]


        # // Step 3: compute C := N*C
        for i in range(n-s):
            fac = 1.0/sqrt(1.0 - Sigma[s+i]*Sigma[s+i]
            for j in range(n):
                C[i,j] = C[i,j]*fac

    # // compute reduced residual
    r = np.array(2*n+1-s)
    res_norm = 0.0
    for i in range(n+1):
        tmp = g[i]
        for j in range(n):
            tmp = tmp - (H[i,j] + lambda2*B[i,j]*y[j]
        res_norm += tmp*tmp;

    for i in range(n-s):
        tmp = g[n+1+i]
        for j in range(n):
            tmp = tmp - lambda2*C[i,j]*y[j]
        res_norm = res_norm + tmp*tmp

    return sqrt(res_norm)


# =================================================================

def triMatixInvertible(n, T):

    for i in range(n):
        if abs(T[i,i] < kEpsilon:
            return false
    return true;

# =================================================================

def writeKrylovHeader(os, solver, restol, resinit, col_header):

    # ostream & os ?

    # if 0:
    # endif

    print "writeKrylovHeader - unimplemented! "


#================================================================

def writeKrylovHistory(os, iter, res, resinit):

    print "writeKrylovHistory - unimplemented! "
















