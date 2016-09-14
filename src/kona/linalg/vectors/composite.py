
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

    init_dual = -0.01

    def __init__(self, primal_vec, dual_vec):
        if isinstance(primal_vec, PrimalVector):
            if not isinstance(dual_vec, DualVectorEQ):
                raise TypeError(
                    'ReducedKKTVector() >> Mismatched dual vector. ' +
                    'Must be DualVectorEQ!')
        elif isinstance(primal_vec, CompositePrimalVector):
            if not isinstance(dual_vec, CompositeDualVector):
                raise TypeError(
                    'ReducedKKTVector() >> Mismatched dual vector. ' +
                    'Must be CompositeDualVector!')
        else:
            raise TypeError(
                'ReducedKKTVector() >> Invalid primal vector. ' +
                'Must be either PrimalVector or CompositePrimalVector!')

        self.primal = primal_vec
        self.dual = dual_vec

        super(ReducedKKTVector, self).__init__(
            [primal_vec, dual_vec])

    def equals_init_guess(self):
        """
        Sets the KKT vector to the initial guess, using the initial design.
        """
        self.primal.equals_init_design()
        self.dual.equals(self.init_dual)

    def equals_KKT_conditions(self, x, state, adjoint, design_work):
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
        design_work : PrimalVector
            Work vector for intermediate calculations.
        """
        # evaluate primal component
        self.primal.equals_lagrangian_total_gradient(
            x.primal, state, x.dual, adjoint, design_work)
        # evaluate multiplier component
        self.dual.equals_constraints(x.primal, state)
        if isinstance(self.primal, CompositePrimalVector):
            self.dual.ineq.minus(self.primal.slack)

class CompositeDualVector(CompositeVector):
    """
        A composite vector representing a combined equality and inequality
        constraints.

        Parameters
        ----------
        _memory : KonaMemory
            All-knowing Kona memory manager.
        eq : DualVectorEQ
            Equality constraints.
        ineq : DualVectorINEQ
            Inequality Constraints
        """

    def __init__(self, dual_eq, dual_ineq):
        if isinstance(dual_eq, DualVectorEQ):
            self.eq = dual_eq
        else:
            raise TypeError('CompositeDualVector() >> ' +
                            'Unidentified equality constraint vector.')

        if isinstance(dual_ineq, DualVectorINEQ):
            self.ineq = dual_ineq
        else:
            raise TypeError('CompositeDualVector() >> ' +
                            'Unidentified inequality constraint vector.')

        super(CompositeDualVector, self).__init__([dual_eq, dual_ineq])

    def equals_constraints(self, at_primal, at_state):
        self.eq.equals_constraints(at_primal.design, at_state)
        self.ineq.equals_constraints(at_primal.design, at_state)

class CompositePrimalVector(CompositeVector):
    """
    A composite vector representing a combined design and slack vectors..

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    design : PrimalVector
        Design component of the composite vector.
    slack : DualVectorINEQ
        Slack components of the composite vector.
    """

    init_slack = 0.0

    def __init__(self, primal_vec, dual_ineq):
        if isinstance(primal_vec, PrimalVector):
            self.design = primal_vec
        else:
            raise TypeError('CompositePrimalVector() >> ' +
                            'Unidentified primal vector.')

        if isinstance(dual_ineq, DualVectorINEQ):
            self.slack = dual_ineq
        else:
            raise TypeError('CompositePrimalVector() >> ' +
                            'Unidentified dual vector.')

        super(CompositePrimalVector, self).__init__([primal_vec, dual_ineq])

    def equals_init_design(self):
        self.design.equals_init_design()
        self.slack.equals(self.init_slack)

    def equals_lagrangian_total_gradient(self, at_primal, at_state,
                                         at_dual, at_adjoint, design_work):
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
        design_work : PrimalVector
            Work vector in the design space.
        """
        # do some aliasing
        at_design = at_primal.design
        at_slack = at_primal.slack
        # compute the design derivative of the lagrangian
        self.design.equals_lagrangian_total_gradient(
            at_design, at_state, at_dual, at_adjoint, design_work)
        # compute the slack derivative of the lagrangian
        self.slack.exp(at_slack)
        self.slack.times(at_dual)
        self.slack.times(-1.)
        self.slack.restrict()

# package imports at the bottom to prevent import errors
from kona.linalg.vectors.common import *