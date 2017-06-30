
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
        else:
            for i in xrange(len(self._vectors)):
                try:
                    self._vectors[i]._check_type(vec._vectors[i])
                except TypeError:
                    raise TypeError("CompositeVector() >> " +
                                    "Mismatched internal vectors!")
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

class PrimalDualVector(CompositeVector):
    """
    A composite vector made up of primal, dual equality, and dual inequality vectors.

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _primal : DesignVector
        Primal component of the composite vector.
    _dual : DualVector
        Dual components of the composite vector.
    """

    init_dual = 0.0  # default initial value for multipliers

    def __init__(self, primal_vec, dual_vec):
        assert isinstance(primal_vec, DesignVector), \
            'PrimalDualVector() >> Mismatched primal vector. ' + \
            'Must be DesignVector!'
        assert isinstance(dual_vec, DualVectorEQ) or \
               isinstance(dual_vec, DualVectorINEQ) or \
               isinstance(dual_vec, CompositeDualVector), \
            'PrimalDualVector() >> Mismatched dual vector. ' + \
            'Must be DualVectorEQ, DualVectorINEQ CompositeDualVector!'

        self.primal = primal_vec
        self.dual = dual_vec

        super(PrimalDualVector, self).__init__([primal_vec, dual_vec])

    def equals_init_guess(self):
        """
        Sets the primal-dual vector to the initial guess, using the initial design.
        """
        self.primal.equals_init_design()
        self.dual.equals(self.init_dual)

    def equals_opt_residual(self, x, state, adjoint, barrier=None,
                            obj_scale=1.0, cnstr_scale=1.0):
        """
        Calculates the following nonlinear vector function:

        .. math::
        r(x,\\lambda_h,\\lambda_g) =
        \\begin{bmatrix}
        \\nabla_x f(x, u) - \\nabla_x h(x, u)^T \\lambda_{h} - \\nabla_x g(x, u)^T \\lambda_{g} \\\\
        h(x,u) \\\\
        -\\theta(|g(x,u) - \\lambda_g|) + \\theta(g(x,u)) + \theta(\\lambda_g)
        \\end{bmatrix}

        where :math:`h(x,u)` are the equality constraints, and :math:`g(x,u)` are the
        inequality constraints.  The vectors :math:`\\lambda_h` and :math:`\\lambda_g`
        are the associated Lagrange multipliers.  The function :math:`\\theta(z)`
        is any strictly increasing function; typically :math:`\\theta(z) = z^3`.  The
        solution to :math:`r(x,\\lambda_h,\\lambda_g) = 0` is equivalent to the
        first-order optimality conditions for the optimization problem

        .. math::
        \\begin{align*}
        \\min_x &f(x,u(x)) \\\\
        \\textsf{s.t.} &h(x,u(x)) = 0, \\\\
        &g(x,u(x)) \geq 0.
        \\end{align*}

        Parameters
        ----------
        x : PrimalDualVector
            Evaluate first-order optimality conditions at this primal-dual point.
        state : StateVector
            Evaluate first-order optimality conditions at this state point.
        adjoint : StateVector
            Evaluate first-order optimality conditions using this adjoint vector.
        obj_scale : float, optional
            Scaling for the objective function.
        cnstr_scale : float, optional
            Scaling for the constraints.
        """
        assert isinstance(x, PrimalDualVector), \
            "PrimalDualVector() >> invalid type x in equals_opt_residual. " + \
            "x vector must be a PrimalDualVector!"
        if isinstance(x.dual, CompositeDualVector):
            dual_eq = x.dual.eq
            dual_ineq = x.dual.ineq
        elif isinstance(x.dual, DualVectorINEQ):
            dual_eq = None
            dual_ineq = x.dual
        elif isinstance(x.dual, DualVectorEQ):
            dual_eq = x.dual
            dual_ineq = None
        else:
            raise AssertionError("PrimalDualVector() >> invalid dual vector type")
        design = x.primal

        design_opt = self.primal

        # first include the objective partial and adjoint contribution
        design_opt.equals_total_gradient(design, state, adjoint, obj_scale)
        if dual_eq is not None:
            # add the Lagrange multiplier products for equality constraints
            design_opt.base.data[:] += design_opt._memory.solver.multiply_dCEQdX_T(
                design.base.data, state.base, dual_eq.base.data) * \
                cnstr_scale
        if dual_ineq is not None:
            # add the Lagrange multiplier products for inequality constraints
            design_opt.base.data[:] += design_opt._memory.solver.multiply_dCINdX_T(
                design.base.data, state.base, dual_ineq.base.data) * \
                cnstr_scale
        # include constraint terms
        self.dual.equals_constraints(design, state, cnstr_scale)
        if isinstance(self.dual, DualVectorINEQ):
            self.dual.equals_mangasarian(self.dual, dual_ineq)
        elif isinstance(self.dual, CompositeDualVector):
            self.dual.ineq.equals_mangasarian(self.dual.ineq, dual_ineq)
        print self.dual.ineq.base.data

class ReducedKKTVector(CompositeVector):
    """
    A composite vector representing a combined primal and dual vectors.

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    _primal : DesignVector or CompositePrimalVector
        Primal component of the composite vector.
    _dual : DualVector
        Dual components of the composite vector.
    """

    init_dual = 0.0

    def __init__(self, primal_vec, dual_vec):
        if isinstance(primal_vec, DesignVector):
            assert isinstance(dual_vec, DualVectorEQ), \
                'ReducedKKTVector() >> Mismatched dual vector. ' + \
                'Must be DualVectorEQ!'
        elif isinstance(primal_vec, CompositePrimalVector):
            assert isinstance(dual_vec, DualVectorINEQ) or \
                   isinstance(dual_vec, CompositeDualVector), \
                'ReducedKKTVector() >> Mismatched dual vector. ' + \
                'Must be DualVecINEQ or CompositeDualVector!'
        else:
            raise AssertionError(
                'ReducedKKTVector() >> Invalid primal vector. ' +
                'Must be either DesignVector or CompositePrimalVector!')

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

    def equals_KKT_conditions(self, x, state, adjoint, barrier=None,
                              obj_scale=1.0, cnstr_scale=1.0):
        """
        Calculates the total derivative of the Lagrangian
        :math:`\\mathcal{L}(x, u) = f(x, u)+ \\lambda_{eq}^T c_{eq}(x, u) + \\lambda_{ineq}^T (c_{ineq}(x, u) - s)` with
        respect to :math:`\\begin{pmatrix}x && s && \\lambda_{eq} && \\lambda_{ineq}\\end{pmatrix}^T`.
        This total derivative represents the Karush-Kuhn-Tucker (KKT)
        convergence conditions for the optimization problem defined by
        :math:`\\mathcal{L}(x, s, \\lambda_{eq}, \\lambda_{ineq})` where the stat variables
        :math:`u(x)` are treated as implicit functions of the design.

        The full expression of the KKT conditions are:

        .. math::
            \\nabla \\mathcal{L} =
            \\begin{bmatrix}
            \\nabla_x f(x, u) + \\nabla_x c_{eq}(x, u)^T \\lambda_{eq} + \\nabla_x c_{inq}(x, u)^T \\lambda_{ineq} \\\\
            \\muS^{-1}e - \\lambda_{ineq} \\\\
            c_{eq}(x, u) \\\\
            c_{ineq}(x, u) - s \\end{bmatrix}

        Parameters
        ----------
        x : ReducedKKTVector
            Evaluate KKT conditions at this primal-dual point.
        state : StateVector
            Evaluate KKT conditions at this state point.
        adjoint : StateVector
            Evaluate KKT conditions using this adjoint vector.
        barrier : float, optional
            Log barrier coefficient for slack variable non-negativity.
        obj_scale : float, optional
            Scaling for the objective function.
        cnstr_scale : float, optional
            Scaling for the constraints.
        """
        # get the design vector
        if isinstance(self.primal, CompositePrimalVector):
            assert isinstance(x.primal, CompositePrimalVector), \
                "ReducedKKTVector() >> KKT point must include slack variables!"
            assert barrier is not None, \
                "ReducedKKTVector() >> Barrier factor must be defined!"
            design = x.primal.design
            self.primal.barrier = barrier
        else:
            assert isinstance(x.primal, DesignVector), \
                "ReducedKKTVector() >> KKT point cannot include slacks!"
            design = x.primal
        dual = x.dual

        # evaluate primal component
        self.primal.equals_lagrangian_total_gradient(
            x.primal, state, dual, adjoint, obj_scale, cnstr_scale)
        # evaluate multiplier component
        self.dual.equals_constraints(design, state, cnstr_scale)
        if isinstance(self.dual, DualVectorINEQ):
            self.dual.minus(x.primal.slack)
        elif isinstance(self.dual, CompositeDualVector):
            self.dual.ineq.minus(x.primal.slack)

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

    def restrict_to_regular(self):
        self.eq.restrict_to_regular()

    def restrict_to_idf(self):
        self.eq.restrict_to_idf()
        self.ineq.equals(0.0)

    def convert_to_design(self, primal_vector):
        self.eq.convert_to_design(primal_vector)

    def equals_constraints(self, at_primal, at_state, scale=1.0):
        """
        Evaluate equality and inequality constraints in-place.

        Parameters
        ----------
        at_primal : DesignVector or CompositePrimalVector
            Primal evaluation point.
        at_state : StateVector
            State evaluation point.
        scale : float, optional
            Scaling for the constraints.
        """
        self.eq.equals_constraints(at_primal, at_state, scale)
        self.ineq.equals_constraints(at_primal, at_state, scale)

class CompositePrimalVector(CompositeVector):
    """
    A composite vector representing a combined design and slack vectors.

    Parameters
    ----------
    _memory : KonaMemory
        All-knowing Kona memory manager.
    design : DesignVector
        Design component of the composite vector.
    slack : DualVectorINEQ
        Slack components of the composite vector.
    """

    init_slack = 1.0

    def __init__(self, primal_vec, dual_ineq):
        if isinstance(primal_vec, DesignVector):
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
        self.barrier = None

    def restrict_to_design(self):
        self.design.restrict_to_design()

    def restrict_to_target(self):
        self.design.restrict_to_target()
        self.slack.equals(0.0)

    def convert_to_dual(self, dual_vector):
        self.design.convert_to_dual(dual_vector)

    def equals_init_design(self):
        self.design.equals_init_design()
        self.slack.equals(self.init_slack)

    def equals_lagrangian_total_gradient(
            self, at_primal, at_state, at_dual, at_adjoint,
            obj_scale=1.0, cnstr_scale=1.0):
        """
        Computes the total primal derivative of the Lagrangian.

        In this case, the primal derivative includes the slack derivative.

        .. math::
            \\nabla_{primal} \\mathcal{L} =
            \\begin{bmatrix}
            \\nabla_x f(x, u) + \\nabla_x c_{eq}(x, u)^T \\lambda_{eq} + \\nabla_x c_{inq}(x, u)^T \\lambda_{ineq} \\\\
            \\muS^{-1}e - \\lambda_{ineq}
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
        obj_scale : float, optional
            Scaling for the objective function.
        cnstr_scale : float, optional
            Scaling for the constraints.
        """
        # make sure the barrier factor is set
        assert self.barrier is not None, \
            "CompositePrimalVector() >> Barrier factor must be set!"
        # do some aliasing
        at_slack = at_primal.slack
        if isinstance(at_dual, CompositeDualVector):
            at_dual_ineq = at_dual.ineq
        else:
            at_dual_ineq = at_dual
        # compute the design derivative of the lagrangian
        self.design.equals_lagrangian_total_gradient(
            at_primal, at_state, at_dual, at_adjoint, obj_scale, cnstr_scale)
        # compute the slack derivative of the lagrangian
        self.slack.equals(at_slack)
        self.slack.pow(-1.)
        self.slack.times(self.barrier)
        self.slack.minus(at_dual_ineq)
        # reset the barrier to None
        self.barrier = None

# package imports at the bottom to prevent import errors
import numpy as np
from kona.linalg.vectors.common import DesignVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
