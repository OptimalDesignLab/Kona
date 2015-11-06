import numpy as np

from kona.linalg.vectors.common import PrimalVector, DualVector

class CompositeVector(object):
    """
    Base class shell for all composite vectors.
    """
    def __init__(self, vectors):
        self._vectors = vectors
        self._memory = self._vectors[0]._memory

    def _check_type(self, vec):
        if not isinstance(vec, type(self)):
            raise TypeError('CompositeVector() >> ' +
                            'Wrong vector type. Must be %s' % type(self))

    def equals(self, rhs):
        """
        Used as the assignment operator.

        If val is a scalar, all vector elements are set to the scalar value.

        If val is a vector, the two vectors are set equal.

        Parameters
        ----------
        rhs : float or CompositeVector
            Right hand side term for assignment.
        """
        if isinstance(rhs,
                      (float, int, np.float64, np.int64, np.float32, np.int32)):
            for i in xrange(len(self._vectors)):
                self._vectors[i].equals(rhs)
        else:
            self._check_type(rhs)
            for i in xrange(len(self._vectors)):
                self._vectors[i].equals(rhs._vectors[i])

    def plus(self, vector):
        """
        Used as the addition operator.

        Adds the incoming vector to the current vector in place.

        Parameters
        ----------
        vector : CompositeVector
            Vector to be added.
        """
        self._check_type(vector)
        for i in xrange(len(self._vectors)):
            self._vectors[i].plus(vector._vectors[i])

    def minus(self, vector):
        """
        Used as the subtraction operator.

        Subtracts the incoming vector from the current vector in place.

        Parameters
        ----------
        vector : CompositeVector
            Vector to be subtracted.
        """
        self._check_type(vector)
        for i in xrange(len(self._vectors)):
            self._vectors[i].minus(vector._vectors[i])

    def times(self, factor):
        """
        Used as the multiplication operator.

        Can multiply with scalars or element-wise with vectors.

        Parameters
        ----------
        factor : float or CompositeVector
            Scalar or vector-valued multiplication factor.
        """
        if isinstance(factor,
                      (float, int, np.float64, np.int64, np.float32, np.int32)):
            for i in xrange(len(self._vectors)):
                self._vectors[i].times(factor)
        else:
            self._check_type(factor)
            for i in xrange(len(self._vectors)):
                self._vectors[i].times(factor._vectors[i])

    def divide_by(self, value):
        """
        Used as the division operator.

        Divides the vector by the given scalar value.

        Parameters
        ----------
        value : float
            Vector to be added.
        """
        if isinstance(value,
                      (float, int, np.float64, np.int64, np.float32, np.int32)):
            for i in xrange(len(self._vectors)):
                self._vectors[i].divide_by(value)
        else:
            raise TypeError(
                'CompositeVector.divide_by() >> Value not a scalar!')

    def equals_ax_p_by(self, a, x, b, y):
        """
        Performs a full a*X + b*Y operation between two vectors, and stores
        the result in place.

        Parameters
        ----------
        a, b : float
            Coefficients for the operation.
        x, y : CompositeVector
            Vectors for the operation
        """
        self._check_type(x)
        self._check_type(y)
        for i in xrange(len(self._vectors)):
            self._vectors[i].equals_ax_p_by(a, x._vectors[i], b, y._vectors[i])

    def inner(self, vector):
        """
        Computes an inner product with another vector.

        Returns
        -------
        float : Inner product.
        """
        self._check_type(vector)
        total_prod = 0.
        for i in xrange(len(self._vectors)):
            total_prod += self._vectors[i].inner(vector._vectors[i])
        return total_prod

    def exp(self, vector):
        """
        Computes the element-wise exponential of the given vector and stores it
        in place.

        Parameters
        ----------
        vector : CompositeVector
        """
        self._check_type(vector)
        for i in xrange(len(self._vectors)):
            self._vectors[i].exp(vector)

    def log(self, vector):
        """
        Computes the element-wise natural log of the given vector and stores it
        in place.

        Parameters
        ----------
        vector : CompositeVector
        """
        self._check_type(vector)
        for i in xrange(len(self._vectors)):
            self._vectors[i].log(vector)

    def pow(self, power):
        """
        Computes the element-wise power of the in-place vector.

        Parameters
        ----------
        power : float
        """
        for i in xrange(len(self._vectors)):
            self._vectors[i].pow(power)

    @property
    def norm2(self):
        """
        Computes the L2 norm of the vector.

        Returns
        -------
        float : L2 norm.
        """
        prod = self.inner(self)
        if prod < 0:
            raise ValueError('CompositeVector.norm2 >> ' +
                             'Inner product is negative!')
        else:
            return np.sqrt(prod)

    @property
    def infty(self):
        """
        Infinity norm of the composite vector.

        Returns
        -------
        float : Infinity norm.
        """
        norms = []
        for i in xrange(len(self._vectors)):
            norms.append(self._vectors[i].infty)
        return max(norms)

class ReducedKKTVector(CompositeVector):
    """
    A composite vector representing a combined primal and dual vectors.

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _primal : PrimalVector or CompositePrimalVector
        Primal component of the composite vector.
    _dual : DualVector
        Dual components of the composite vector.
    """

    def __init__(self, primal_vec, dual_vec):
        if isinstance(primal_vec, PrimalVector) or \
           isinstance(primal_vec, CompositePrimalVector):
            self._primal = primal_vec
        else:
            raise TypeError('CompositeVector() >> ' +
                            'Unidentified primal vector.')

        if isinstance(dual_vec, DualVector):
            self._dual = dual_vec
        else:
            raise TypeError('CompositeVector() >> ' +
                            'Unidentified dual vector.')

        super(ReducedKKTVector, self).__init__([primal_vec, dual_vec])

    def equals_init_guess(self):
        """
        Sets the KKT vector to the initial guess, using the initial design.
        """
        self._primal.equals_init_design()
        self._dual.equals(-1.0)

    def equals_KKT_conditions(self, x, state, adjoint, primal_work, dual_work):
        """
        Calculates the total derivative of the Lagrangian
        :math:`\\mathcal{L}(x, u) = f(x, u)+ \\lambda^T (c(x, u) - e^s)` with
        respect to :math:`\\begin{pmatrix}x && s && \\lambda\\end{pmatrix}^T`.
        This total derivative represents the Karush-Kuhn-Tucker (KKT)
        convergence conditions for the optimization problem defined by
        :math:`\\mathcal{L}(x, s, \\lambda)` where the stat variables
        :math:`u(x)` are treated as implicit functions of the design.

        The full expression of the KKT conditions are:

        .. math::
            \\nabla \\mathcal{L} =
            \\begin{bmatrix}
            \\nabla_x f(x, u) + \\nabla_x c(x, u)^T \\lambda \\\\
            -\\lambda^T e^s \\\\
            c(x, u) - e^s \\end{bmatrix}

        Parameters
        ----------
        x : ReducedKKTVector
            Evaluate KKT conditions at this primal-dual point.
        state : StateVector
            Evaluate KKT conditions at this state point.
        adjoint : StateVector
            Evaluate KKT conditions using this adjoint vector.
        slack : DualVector
            Evaluate KKT conditions using these slack variables.
        primal_work : PrimalVector
            Work vector for intermediate calculations.
        """
        # evaluate primal component
        self._primal.equals_lagrangian_total_gradient(
            x._primal, state, x._dual, adjoint, primal_work)
        # evaluate multiplier component
        if isinstance(self._primal, CompositePrimalVector):
            self._dual.equals_constraints(x._primal._design, state)
            dual_work.exp(x._primal._slack)
            dual_work.times(-1.)
            dual_work.restrict()
            self._dual.plus(dual_work)
        else:
            self._dual.equals_constraints(x._primal, state)

class CompositePrimalVector(CompositeVector):
    """
    A composite vector representing a combined design and slack vectors..

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _primal : PrimalVector
        Design component of the composite vector.
    _slack : DualVector
        Slack components of the composite vector.
    """

    def __init__(self, primal_vec, dual_vec):
        if isinstance(primal_vec, PrimalVector):
            self._design = primal_vec
        else:
            raise TypeError('CompositeVector() >> ' +
                            'Unidentified primal vector.')

        if isinstance(dual_vec, DualVector):
            self._slack = dual_vec
        else:
            raise TypeError('CompositeVector() >> ' +
                            'Unidentified dual vector.')

        super(CompositePrimalVector, self).__init__([primal_vec, dual_vec])

    def equals_init_design(self):
        self._design.equals_init_design()
        self._slack.equals(0.0)

    def equals_lagrangian_total_gradient(self, at_primal, at_state,
                                         at_dual, at_adjoint, primal_work):
        """
        Computes the total primal derivative of the Lagrangian.

        In this case, the primal derivative includes the slack derivative.

        .. math::
            \\nabla_{primal} \\mathcal{L} =
            \\begin{bmatrix}
            \\nabla_x f(x, u) + \\nabla_x c(x, u)^T \\lambda \\\\
            -\\lambda^T e^s
            \\end{bmatrix}

        Parameters
        ----------
        at_primal : CompositePrimalVector
            The design/slack vector at which the derivative is computed.
        at_state : StateVector
            State variables at which the derivative is computed.
        at_dual : DualVector
            Lagrange multipliers at which the derivative is computed.
        at_adjoint : StateVector
            Pre-computed adjoint variables for the Lagrangian.
        primal_work : PrimalVector
            Work vector in the primal space.
        """
        # do some aliasing
        at_design = at_primal._design
        at_slack = at_primal._slack
        # compute the design derivative of the lagrangian
        self._design.equals_lagrangian_total_gradient(
            at_design, at_state, at_dual, at_adjoint, primal_work)
        # compute the slack derivative of the lagrangian
        self._slack.exp(at_slack)
        self._slack.times(at_dual)
        self._slack.times(-1.)
        self._slack.restrict()
