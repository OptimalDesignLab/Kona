
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
    primal : DesignVector
        Primal component of the composite vector.
    eq : DualVectorEQ
        Dual component corresponding to the equality constraints.
    ineq: DualVectorINEQ
        Dual component corresponding to the inequality constraints.
    """

    init_dual = 0.0  # default initial value for multipliers

    def __init__(self, primal_vec, eq_vec=None, ineq_vec=None):
        assert isinstance(primal_vec, DesignVector), \
            'PrimalDualVector() >> primal_vec must be a DesignVector!'
        if eq_vec is not None:
            assert isinstance(eq_vec, DualVectorEQ), \
                'PrimalDualVector() >> eq_vec must be a DualVectorEQ!'
        if ineq_vec is not None:
            assert isinstance(ineq_vec, DualVectorINEQ), \
                'PrimalDualVector() >> ineq_vec must be a DualVectorINEQ!'
        self.primal = primal_vec
        self.eq = eq_vec
        self.ineq = ineq_vec
        vec_list = [primal_vec, eq_vec, ineq_vec]
        super(PrimalDualVector, self).__init__([vec for vec in vec_list if vec is not None])

    def get_num_var(self):
        """
        Returns the total number of variables in the PrimalDualVector
        """
        nvar = self.primal._memory.ndv
        if self.eq is not None:
            nvar += self.eq._memory.neq
        if self.ineq is not None:
            nvar += self.ineq._memory.nineq
        return nvar

    def get_dual(self):
        """
        Returns 1) eq or ineq if only one is None, 2) a CompositeDualVector if neither is none or
        3) None if both are none
        """
        dual = None
        if self.eq is not None and self.ineq is not None:
            dual = CompositeDualVector(self.eq, self.ineq)
        elif self.eq is not None:
            dual = self.eq
        elif self.ineq is not None:
            dual = self.ineq
        return dual

    def equals_init_guess(self):
        """
        Sets the primal-dual vector to the initial guess, using the initial design.
        """
        self.primal.equals_init_design()
        if self.eq is not None:
            self.eq.equals(self.init_dual)
        if self.ineq is not None:
            self.ineq.equals(self.init_dual)

    def equals_KKT_conditions(self, x, state, adjoint, obj_scale=1.0, cnstr_scale=1.0):
        """
        Calculates the total derivative of the Lagrangian
        :math:`\\mathcal{L}(x, u) = f(x, u)+ \\lambda_{h}^T h(x, u) + \\lambda_{g}^T g(x, u)` with
        respect to :math:`\\begin{pmatrix}x && \\lambda_{h} && \\lambda_{g}\\end{pmatrix}^T`,
        where :math:`h` denotes the equality constraints (if any) and :math:`g` denotes
        the inequality constraints (if any).  Note that these (total) derivatives do not
        represent the complete set of first-order optimality conditions in the case of
        inequality constraints.

        Parameters
        ----------
        x : PrimalDualVector
            Evaluate derivatives at this primal-dual point.
        state : StateVector
            Evaluate derivatives at this state point.
        adjoint : StateVector
            Evaluate derivatives using this adjoint vector.
        obj_scale : float, optional
            Scaling for the objective function.
        cnstr_scale : float, optional
            Scaling for the constraints.
        """
        assert isinstance(x, PrimalDualVector), \
            "PrimalDualVector() >> invalid type x in equals_opt_residual. " + \
            "x vector must be a PrimalDualVector!"
        if x.eq is None or self.eq is None:
            assert self.eq is None and x.eq is None, \
                "PrimalDualVector() >> inconsistency with x.eq and self.eq!"
        if x.ineq is None or self.ineq is None:
            assert self.ineq is None and x.ineq is None, \
                "PrimalDualVector() >> inconsistency with x.ineq and self.ineq!"
        # set some aliases
        design = x.primal
        dLdx = self.primal
        ceq = self.eq
        cineq = self.ineq
        # first include the objective partial and adjoint contribution
        dLdx.equals_total_gradient(design, state, adjoint, obj_scale)
        if x.eq is not None:
            # subtract the Lagrange multiplier products for equality constraints
            dLdx.base.data[:] -= dLdx._memory.solver.multiply_dCEQdX_T(
                design.base.data, state.base, x.eq.base.data) * \
                cnstr_scale
        if x.ineq is not None:
            # subract the Lagrange multiplier products for inequality constraints
            dLdx.base.data[:] -= dLdx._memory.solver.multiply_dCINdX_T(
                design.base.data, state.base, x.ineq.base.data) * \
                cnstr_scale
        # include constraint terms
        if ceq is not None:
            ceq.equals_constraints(design, state, cnstr_scale)
        if cineq is not None:
            cineq.equals_constraints(design, state ,cnstr_scale)

    def get_optimality_and_feasiblity(self):
        """
        Returns the norm of the primal (opt) the dual parts of the vector (feas).  If the dual
        parts of the vector are both None, then feas is returned as zero.
        """
        opt = self.primal.norm2
        feas = 0.0
        if self.eq is not None:
            feas += self.eq.norm2**2
        if self.ineq is not None:
            feas += self.ineq.norm2**2
        feas = np.sqrt(feas)
        return opt, feas

    def equals_primaldual_residual(self, dLdx, ineq_mult=None):
        """
        Using dLdx=:math:`\\begin{pmatrix} \\nabla_x L && h && g \\end{pmatrix}`, which can be
        obtained from the method equals_KKT_conditions (as well as the inequality multiplier when
        inequality constraints are present) this method computes the following nonlinear vector
        function:

        .. math::
        r(x,\\lambda_h,\\lambda_g;\\mu) =
        \\begin{bmatrix}
        \\left[\\nabla_x f(x, u) - \\nabla_x h(x, u)^T \\lambda_{h} - \\nabla_x g(x, u)^T \\lambda_{g}\\right] \\\\
        -h(x,u) \\\\
        -|g(x,u) - \\lambda_g| + g(x,u) + \\lambda_g
        \\end{bmatrix}

        where :math:`h(x,u)` are the equality constraints, and :math:`g(x,u)` are the
        inequality constraints.  The vectors :math:`\\lambda_h` and :math:`\\lambda_g`
        are the associated Lagrange multipliers.

        Parameters
        ----------
        dLdx : PrimalDualVector
            The total derivative of the Lagranginan with respect to the primal and dual variables.
        ineq_mult : DualVectorINEQ, optional
            The current multiplier associated with the inequality constraints.
        """
        assert isinstance(dLdx, PrimalDualVector), \
            "PrimalDualVector() >> dLdx must be a PrimalDualVector!"
        if dLdx.ineq is not None or ineq_mult is not None:
            assert isinstance(ineq_mult, DualVectorINEQ), \
                "PrimalDualVector() >> ineq_mult, if present, must be a DualVectorINEQ!"
            assert dLdx.ineq is not None and ineq_mult is not None, \
                "PrimalDualVector() >> dLdx.ineq and ineq_mult are inconsistent!"
        if dLdx == self:
            "PrimalDualVector() >> equals_primaldual_residual is not in-place!"
        # construct the primal part of the residual: dLdx
        self.primal.equals(dLdx.primal)
        if dLdx.eq is not None:
            # include the equality constraint part: -h
            self.eq.equals(dLdx.eq)
            self.eq.times(-1.)
        if dLdx.ineq is not None:
            # include the inequality constraint part
            self.ineq.equals_mangasarian(dLdx.ineq, ineq_mult)

    def equals_homotopy_residual(self, dLdx, x, init, mu=1.0):
        """
        Using dLdx=:math:`\\begin{pmatrix} \\nabla_x L && h && g \\end{pmatrix}`, which can be
        obtained from the method equals_KKT_conditions, as well as the initial values
        init=:math:`\\begin{pmatrix} x_0 && h(x_0,u_0) && g(x_0,u_0) \\end{pmatrix}` and the
        current point x=:math:`\\begin{pmatrix} x && \lambda_h && \lambda_g \end{pmatrix}`, this
        method computes the following nonlinear vector function:

        .. math::
        r(x,\\lambda_h,\\lambda_g;\\mu) =
        \\begin{bmatrix}
        \\mu\\left[\\nabla_x f(x, u) - \\nabla_x h(x, u)^T \\lambda_{h} - \\nabla_x g(x, u)^T \\lambda_{g}\\right] + (1 - \\mu)(x - x_0) \\\\
        -\\mu h(x,u) - (1-\mu)\\lambda_h \\\\
        -|g(x,u) - (1-\\mu)*g_0 - \\lambda_g|^3 + (g(x,u) - (1-\\mu)g_0)^3 + \\lambda_g^3 - (1-\\mu)\\hat{g}
        \\end{bmatrix}

        where :math:`h(x,u)` are the equality constraints, and :math:`g(x,u)` are the
        inequality constraints.  The vectors :math:`\\lambda_h` and :math:`\\lambda_g`
        are the associated Lagrange multipliers.  When mu=1.0, we recover a set of nonlinear
        algebraic equations equivalent to the first-order optimality conditions.

        Parameters
        ----------
        dLdx : PrimalDualVector
            The total derivative of the Lagranginan with respect to the primal and dual variables.
        x : PrimalDualVector
            The current solution vector value corresponding to dLdx.
        init: PrimalDualVector
            The initial primal variable, as well as the initial constraint values.
        mu : float
            Homotopy parameter; must be between 0 and 1.
        """
        assert isinstance(dLdx, PrimalDualVector), \
            "PrimalDualVector() >> dLdx must be a PrimalDualVector!"
        assert isinstance(x, PrimalDualVector), \
            "PrimalDualVector() >> x must be a PrimalDualVector!"
        assert isinstance(init, PrimalDualVector), \
            "PrimalDualVector() >> init must be a PrimalDualVector!"
        if dLdx.eq is None or self.eq is None or x.eq is None or init.eq is None:
            assert dLdx.eq is None and self.eq is None and x.eq is None and init.eq is None, \
                "PrimalDualVector() >> inconsistent eq component in self, dLdx, x, and/or init!"
        if dLdx.ineq is None or self.ineq is None or x.ineq is None or init.ineq is None:
            assert dLdx.ineq is None and self.ineq is None and x.ineq is None \
                and init.ineq is None, \
                "PrimalDualVector() >> inconsistent ineq component in self, dLdx, x, and/or init!"
        if dLdx == self or x == self or init == self:
            "PrimalDualVector() >> equals_homotopy_residual is not in-place!"
        # construct the primal part of the residual: mu*dLdx + (1-mu)(x - x_0)
        self.primal.equals_ax_p_by(1.0, x.primal, -1.0, init.primal)
        self.primal.equals_ax_p_by(mu, dLdx.primal, (1.0 - mu), self.primal)
        if dLdx.eq is not None:
            # include the equality constraint part: -mu*h - (1-mu)*lambda_h
            self.eq.equals_ax_p_by(-mu, dLdx.eq, -(1.0 - mu), x.eq)
            #self.eq.equals_ax_p_by(mu, dLdx.eq, -(1.0 - mu), x.eq)
        if dLdx.ineq is not None:
            # include the inequality constraint part
            self.ineq.equals_ax_p_by(1.0, dLdx.ineq, -(1.0 - mu), init.ineq)
            self.ineq.equals_mangasarian(self.ineq, x.ineq)
            self.ineq.minus((1.0 - mu)*0.1)

    def equals_projgrad_residual(self, dLdx, x, init, mu=1.0):
        # construct the primal part of the residual: mu*dLdx + (1-mu)(x - x_0)
        self.primal.equals_ax_p_by(1.0, x.primal, -1.0, init.primal)
        self.primal.equals_ax_p_by(mu, dLdx.primal, (1.0 - mu), self.primal)
        if dLdx.eq is not None:
            # include the equality constraint part: -mu*h - (1-mu)*lambda_h
            self.eq.equals_ax_p_by(-mu, dLdx.eq, -(1.0 - mu), x.eq)
            #self.eq.equals_ax_p_by(mu, dLdx.eq, -(1.0 - mu), x.eq)
        if dLdx.ineq is not None:
            # include the inequality constraint part
            self.ineq.equals_ax_p_by(1.0, dLdx.ineq, -(1.0 - mu), init.ineq)
            self.ineq.equals_mangasarian(self.ineq, x.ineq)
            self.ineq.minus((1.0 - mu)*0.1)

    def equals_predictor_rhs(self, dLdx, x, init, mu=1.0):
        """
        Using dLdx=:math:`\\begin{pmatrix} \\nabla_x L && h && g \\end{pmatrix}`, which can be
        obtained from the method equals_KKT_conditions, as well as the initial values
        init=:math:`\\begin{pmatrix} x_0 && h(x_0,u_0) && g(x_0,u_0) \\end{pmatrix}` and the
        current point x=:math:`\\begin{pmatrix} x && \lambda_h && \lambda_g \end{pmatrix}`, this
        method computes the right-hand-side for the homotopy-predictor step, that is

        .. math::
        \partial r/\partial \\mu =
        \\begin{bmatrix}
        \\left[\\nabla_x f(x, u) - \\nabla_x h(x, u)^T \\lambda_{h} - \\nabla_x g(x, u)^T \\lambda_{g}\\right] - (x - x_0) \\\\
        -h(x,u) + \\lambda_h \\\\
        -3*(g_0)|g(x,u) - (1-\\mu)*g_0 - \\lambda_g|^2 + 3*g_0*(g(x,u) - (1-\\mu)g_0)^2 + \hat{g}
        \\end{bmatrix}

        where :math:`h(x,u)` are the equality constraints, and :math:`g(x,u)` are the
        inequality constraints.  The vectors :math:`\\lambda_h` and :math:`\\lambda_g`
        are the associated Lagrange multipliers.

        Parameters
        ----------
        dLdx : PrimalDualVector
            The total derivative of the Lagranginan with respect to the primal and dual variables.
        x : PrimalDualVector
            The current solution vector value corresponding to dLdx.
        init: PrimalDualVector
            The initial primal variable, as well as the initial constraint values.
        mu : float
            Homotopy parameter; must be between 0 and 1.
        """
        assert isinstance(dLdx, PrimalDualVector), \
            "PrimalDualVector() >> dLdx must be a PrimalDualVector!"
        assert isinstance(init, PrimalDualVector), \
            "PrimalDualVector() >> init must be a PrimalDualVector!"
        if dLdx.eq is None or self.eq is None or x.eq is None or init.eq is None:
            assert dLdx.eq is None and self.eq is None and x.eq is None and init.eq is None, \
                "PrimalDualVector() >> inconsistent eq component in self, dLdx, x, and/or init!"
        if dLdx.ineq is None or self.ineq is None or x.ineq is None or init.ineq is None:
            assert dLdx.ineq is None and self.ineq is None and x.ineq is None \
                and init.ineq is None, \
                "PrimalDualVector() >> inconsistent ineq component in self, dLdx, x, and/or init!"
        if dLdx == self or x == self or init == self:
            "PrimalDualVector() >> equals_predictor_rhs is not in-place!"
        # construct the primal part of the rhs: dLdx - (x - x_0)
        self.primal.equals_ax_p_by(1.0, dLdx.primal, -1.0, x.primal)
        self.primal.plus(init.primal)
        if dLdx.eq is not None:
            # include the equality constraint part: -h + lambda_h
            self.eq.equals_ax_p_by(-1., dLdx.eq, 1.0, x.eq)
            #self.eq.equals_ax_p_by(1., dLdx.eq, 1.0, x.eq)
        if dLdx.ineq is not None:
            # include the inequality constraint part
            self.ineq.equals_ax_p_by(1.0, dLdx.ineq, -(1.0 - mu), init.ineq)
            self.ineq.deriv_mangasarian(self.ineq, x.ineq)
            self.ineq.times(init.ineq)
            self.ineq.plus(0.1)

    def get_base_data(self, A):
        """
        Inserts the PrimalDualVector's underlying data into the given array

        Parameters
        ----------
        A : numpy array
            Array into which data is inserted.
        """
        ptr = 0
        A[ptr:ptr+self.primal._memory.ndv] = self.primal.base.data[:]
        ptr += self.primal._memory.ndv
        if self.eq is not None:
            A[ptr:ptr+self.eq._memory.neq] = self.eq.base.data[:]
            ptr += self.eq._memory.neq
        if self.ineq is not None:
            A[ptr:ptr+self.ineq._memory.nineq] = self.ineq.base.data[:]

    def set_base_data(self, A):
        """
        Copies the given array into the PrimalDualVector's underlying data

        Parameters
        ----------
        A : numpy array
            Array that is copied into the PrimalDualVector.
        """
        ptr = 0
        self.primal.base.data[:] = A[ptr:ptr+self.primal._memory.ndv]
        ptr += self.primal._memory.ndv
        if self.eq is not None:
            self.eq.base.data[:] = A[ptr:ptr+self.eq._memory.neq]
            ptr += self.eq._memory.neq
        if self.ineq is not None:
            self.ineq.base.data[:] = A[ptr:ptr+self.ineq._memory.nineq]

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
