from kona.algorithms.base_algorithm import OptimizationAlgorithm
import pdb, pickle

"""
if self.ineq_factory is not None:
    if self.eq_factory is not None: 
        # Both Inequality and Equality 
    else:
        # Inequality Only
else:
    # Equality Only

Unconstrained Case is NOT considered
"""


class PredictorCorrectorCnstrCond(OptimizationAlgorithm):

    def __init__(self, primal_factory, state_factory,
                 eq_factory, ineq_factory, optns=None):
        # trigger base class initialization
        super(PredictorCorrectorCnstrCond, self).__init__(
            primal_factory, state_factory, eq_factory, ineq_factory, optns
        )

        # number of vectors required in solve() method  + fgmres
        krylov_size = get_opt(self.optns, 10, 'rsnk', 'subspace_size')
        self.primal_factory.request_num_vectors(40 + 2*krylov_size)
        self.state_factory.request_num_vectors(10)

        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(40 + 2*krylov_size)

        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(80 + 4*krylov_size)

        # general options
        ############################################################
        self.factor_matrices = get_opt(self.optns, False, 'matrix_explicit')

        # reduced hessian settings
        ############################################################
        self.symmetric = get_opt(self.optns, False, 'symmetric')

        kkt_optns = {
            'symmetric'   : get_opt(self.optns, False, 'symmetric'), 
            'product_fac' : get_opt(self.optns, 0.001, 'rsnk', 'product_fac'),
            'lambda'      : get_opt(self.optns, 0.0, 'rsnk', 'lambda' ),
            'scale'       : get_opt(self.optns, 1.0, 'rsnk', 'scale' ),
            'dynamic_tol'  : get_opt(self.optns, False,'rsnk', 'dynamic_tol'),  
        }

        self.hessian = ReducedKKTMatrix(
            [self.primal_factory, self.state_factory, self.eq_factory, self.ineq_factory], kkt_optns)
        self.mat_vec = self.hessian.product

        # hessian preconditiner settings
        ############################################################
        self.precond = get_opt(self.optns, None, 'rsnk', 'precond')
        
      
        self.svd_inequ = None
        self.svd_general = None                   

        if self.precond is 'svd_inequ': 
            print 'svd_inequ is used! '
            svd_optns = {
                'bfgs_max_stored' : get_opt(self.optns, 10, 'svd', 'bfgs_max_stored'),
                'lanczos_size'    : get_opt(self.optns, 20, 'svd', 'lanczos_size'),
                'mu_exact'        : get_opt(self.optns, -1.0, 'svd', 'mu_exact'),
                'sig_exact'        : get_opt(self.optns, 1.0, 'svd', 'sig_exact'),
                'beta'            : get_opt(self.optns, 1.0, 'svd', 'beta'), 
                'cmin'            : get_opt(self.optns, 1e-3, 'svd', 'cmin'), 
                'fstopo'          : get_opt(self.optns, False, 'svd', 'fstopo'),
            }
            self.svd_inequ = SvdInequ(
                [primal_factory, state_factory, eq_factory, ineq_factory], svd_optns)
            self.precond = self.svd_inequ.solve    

        elif self.precond is 'svd_general': 
            # print 'svd_gen for both equality and inequality is used! '
            svd_optns = {
                'bfgs_max_stored' : get_opt(self.optns, 10, 'svd', 'bfgs_max_stored'),
                'lanczos_size'    : get_opt(self.optns, 20, 'svd', 'lanczos_size'),
                'mu_exact'        : get_opt(self.optns, -1.0, 'svd', 'mu_exact'),
                'sig_exact'       : get_opt(self.optns, 1.0, 'svd', 'sig_exact'),
                'beta'            : get_opt(self.optns, 1.0, 'svd', 'beta'), 
                'mu_min'          : get_opt(self.optns, 1e-3, 'svd', 'mu_min'), 
            }
            self.svd_gen = SvdGen(
                [primal_factory, state_factory, eq_factory, ineq_factory], svd_optns)
            self.precond = self.svd_gen.solve                

        else:
            self.eye = IdentityMatrix()
            self.precond = self.eye.product

        self.eye = IdentityMatrix()
        self.eye_precond = self.eye.product

        # krylov solver settings
        ############################################################
        krylov_optns = {
            'krylov_file'   : get_opt(
                self.optns, 'kona_krylov.dat', 'rsnk', 'krylov_file'),
            'subspace_size' : get_opt(self.optns, 10, 'rsnk', 'subspace_size'),
            'check_res'     : get_opt(self.optns, True, 'rsnk', 'check_res'),
            'rel_tol'       : get_opt(self.optns, 1e-2, 'rsnk', 'rel_tol'),
        }
        self.krylov = FGMRES(self.primal_factory, krylov_optns,
                             eq_factory=self.eq_factory, ineq_factory=self.ineq_factory)

        self.outdir = krylov_optns['krylov_file'].split('/')[1]
        # homotopy options
        ############################################################
        self.mu = get_opt(self.optns, 1.0, 'homotopy', 'init_homotopy_parameter')
        self.inner_tol = get_opt(self.optns, 1e-1, 'homotopy', 'inner_tol')
        self.inner_maxiter = get_opt(self.optns, 50, 'homotopy', 'inner_maxiter')
        self.step = get_opt(
            self.optns, 0.05, 'homotopy', 'init_step')
        self.nom_dcurve = get_opt(self.optns, 1.0, 'homotopy', 'nominal_dist')
        self.nom_angl = get_opt(
            self.optns, 5.0*np.pi/180., 'homotopy', 'nominal_angle')
        self.max_factor = get_opt(self.optns, 2.0, 'homotopy', 'max_factor')
        self.min_factor = get_opt(self.optns, 0.5, 'homotopy', 'min_factor')
        self.dmu_max = get_opt(self.optns, -0.1, 'homotopy', 'dmu_max')
        self.dmu_min = get_opt(self.optns, -0.9, 'homotopy', 'dmu_min')
        self.mu_correction = get_opt(self.optns, 1.0, 'homotopy', 'mu_correction')
        self.precond_on_mu = get_opt(self.optns, 1.0, 'homotopy', 'mu_pc_on')
        

    def _write_header(self, opt_tol, feas_tol):
        self.hist_file.write(
            '# Kona Param. Contn. convergence history ' +
            '(opt tol = %e | feas tol = %e)\n'%(opt_tol, feas_tol) +
            '# outer' + ' '*5 +
            'inner' + ' '*5 +
            ' cost' + ' '*5 +
            ' objective' + ' ' * 5 +
            'lagrangian' + ' ' * 5 +
            '  opt 2norm' + ' '*5 +
            'S*Lam 2norm' + ' '*5 +
            ' feas 2norm' + ' '*5 +
            '  homotopy' + ' ' * 5 +
            '   hom opt' + ' '*5 +
            '  hom feas' + ' '*5 +
            'mu        ' + ' '*5 +
            '  opt infty' + ' '*5 +
            'S*Lam infty' + ' '*5 +
            ' feas infty' + ' '*5 +
            '\n'
        )

    def _write_outer(self, outer, cost, obj, lag, opt_norm, slam_norm, feas_norm, mu, opt_inf, slam_inf, feas_inf):
        dummy0 = 0.0
        dummy_fmt = '%.4e'%dummy0

        if obj < 0.:
            obj_fmt = '%.3e'%obj
        else:
            obj_fmt = ' %.3e'%obj
        if lag < 0.:
            lag_fmt = '%.3e'%lag
        else:
            lag_fmt = ' %.3e'%lag
        self.hist_file.write(
            '%7i' % outer + ' ' * 5 +
            '%5i' % dummy0  + ' ' * 5 +
            '%5i' % cost + ' ' * 5 +
            obj_fmt + ' ' * 5 +
            lag_fmt + ' ' * 5 +
            '%.4e' % opt_norm + ' ' * 5 +
            '%.4e' % slam_norm + ' ' * 5 +
            '%.4e' % feas_norm + ' ' * 5 +
            dummy_fmt + ' ' * 5 +
            dummy_fmt + ' ' * 5 +
            dummy_fmt + ' ' * 5 +
            '%1.4f' % mu + ' ' * 10 +  
            '%.4e' % opt_inf + ' ' * 5 +
            '%.4e' % slam_inf + ' ' * 5 +
            '%.4e' % feas_inf + ' ' * 5 +            
            '\n'
        )

    def _write_inner(self, outer, inner,
                     obj, lag, opt_norm, slam_norm, feas_norm,
                     hom, hom_opt, hom_feas, opt_inf, slam_inf, feas_inf):          
        if obj < 0.:
            obj_fmt = '%.3e'%obj
        else:
            obj_fmt = ' %.3e'%obj
        if lag < 0.:
            lag_fmt = '%.3e'%lag
        else:
            lag_fmt = ' %.3e'%lag
        if hom < 0.:
            hom_fmt = '%.3e'%hom
        else:
            hom_fmt = ' %.3e'%hom
        self.hist_file.write(
            '%7i' % outer + ' ' * 5 +
            '%5i' % inner + ' ' * 5 +
            '%5i' % self.primal_factory._memory.cost + ' ' * 5 +
            obj_fmt + ' ' * 5 +
            lag_fmt + ' ' * 5 +
            '%.4e' % opt_norm + ' ' * 5 +
            '%.4e' % slam_norm + ' ' * 5 +
            '%.4e' % feas_norm + ' ' * 5 +
            hom_fmt + ' ' * 5 +
            '%.4e' % hom_opt + ' ' * 5 +
            '%.4e' % hom_feas + ' ' * 5 +
            '%.6e' % self.mu + ' ' * 10 +    
            '%.4e' % opt_inf + ' ' * 5 +
            '%.4e' % slam_inf + ' ' * 5 +
            '%.4e' % feas_inf + ' ' * 5 +                 
            '\n'
        )

    def _generate_primal(self):
        if self.ineq_factory is None:
            return self.primal_factory.generate()
        else:
            prim = self.primal_factory.generate()
            dual_ineq = self.ineq_factory.generate()        
            return CompositePrimalVector(prim, dual_ineq)

    def _generate_dual(self):

        if self.ineq_factory is not None:
            if self.eq_factory is not None:
                dual_eq = self.eq_factory.generate()
                dual_ineq = self.ineq_factory.generate()
                out = CompositeDualVector(dual_eq, dual_ineq)
            else:    
                out = self.ineq_factory.generate()
        else:
            out = self.eq_factory.generate()

        return  out

    def _generate_kkt(self):
        primal = self._generate_primal()
        dual = self._generate_dual()
        return ReducedKKTVector(primal, dual)

    def _mat_vec(self, in_vec, out_vec):

        self.hessian.product(in_vec, out_vec)
        out_vec.times(1. - self.mu)

        self.prod_work.equals(in_vec)
        self.prod_work.times(self.mu)

        if self.symmetric is True: 
            self.prod_work.primal.slack.times(self.current_x.primal.slack)

        out_vec.primal.plus(self.prod_work.primal)
        out_vec.dual.minus(self.prod_work.dual)

    def find_step(self, max_mu_step, x, t):
            # --------- fraction to boundary rule ----------

            tau_s = 1e-6

            slack_steps = (tau_s - x.primal.slack.base.data)/t.primal.slack.base.data

            if any(slack_steps > 0):   # step limit should be used, as some t_s < 0
                max_slack_step = min(slack_steps[slack_steps > 0])
            else:
                max_slack_step = max_mu_step

            ind_active_s0 = np.where( x.primal.slack.base.data <= tau_s)
           
            return min(max_mu_step, max_slack_step), ind_active_s0

            
    def solve(self):
        self.info_file.write(
            '\n' +
            '**************************************************\n' +
            '***        Using Parameter Continuation        ***\n' +
            '**************************************************\n' +
            '\n')

        # get the vectors we need
        x = self._generate_kkt()
        x0 = self._generate_kkt()
        x_save = self._generate_kkt()
        x_temp = self._generate_kkt()
        dJdX = self._generate_kkt()
        X_oldx = self._generate_kkt()

        dJdX_save = self._generate_kkt()
        dJdX_hom = self._generate_kkt()
        dx = self._generate_kkt()
        dx_newt = self._generate_kkt()
        dx_bfgs = self._generate_kkt()
        rhs_vec = self._generate_kkt()
        t = self._generate_kkt()
        t_save = self._generate_kkt()
        self.prod_work = self._generate_kkt()

        self.current_x = self._generate_kkt()
        self.current_dldx = self._generate_kkt()
        dldx_bfgs = self._generate_kkt()
        kkt_work = self._generate_kkt()

        state_old = self.state_factory.generate()
        adj_old = self.state_factory.generate()
        state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        state_save = self.state_factory.generate()
        state_work_svd = self.state_factory.generate()
        adj = self.state_factory.generate()
        adj_save = self.state_factory.generate()
        adj_work = self.state_factory.generate()

        primal_work = self._generate_primal()
        design_work = self.primal_factory.generate()
        dual_work = self._generate_dual()
        dual_work2 = self._generate_dual()

        if self.svd_inequ is not None:
            X_olddualS = self._generate_kkt()
            dLdX_olddualS = self._generate_kkt()
            old_x = self._generate_kkt()
            old_dldx = self._generate_kkt()


        if self.ineq_factory is not None:
            if self.eq_factory is not None: 
                self.info_file.write(
                    '# of design vars = %i\n' % len(x.primal.design.base.data) +
                    '# of slack vars  = %i\n' % len(x.dual.ineq.base.data) +
                    '# of ineq cnstr    = %i\n' % len(x.dual.ineq.base.data) +
                    '# of eq cnstr    = %i\n' % len(x.dual.eq.base.data) +
                    '\n'
                ) 
            else:
                self.info_file.write(
                    '# of design vars = %i\n' % len(x.primal.design.base.data) +
                    '# of slack vars  = %i\n' % len(x.dual.base.data) +
                    '# of ineq cnstr    = %i\n' % len(x.dual.base.data) +
                    '\n'
                )
        else:
            self.info_file.write(
                '# of design vars = %i\n' % len(x.primal.base.data) +
                '# of eq cnstr    = %i\n' % len(x.dual.base.data) +
                '\n'
            )


        EPS = 1e-9    # np.finfo(np.float64).eps
        # initialize the problem at the starting point
        x0.equals_init_guess()
        if self.ineq_factory is not None:
            x0.primal.slack.base.data[x0.primal.slack.base.data < 1e-6] = 1e-6
        x.equals(x0)
        
        if not state.equals_primal_solution(x.primal):
            raise RuntimeError('Invalid initial point! State-solve failed.')
        if self.factor_matrices:
            factor_linear_system(x.primal, state)

        obj_fac = 1.
        cnstr_fac = 1.

        # compute the lagrangian adjoint
        adj.equals_lagrangian_adjoint(
            x, state, state_work, obj_scale=obj_fac, cnstr_scale=cnstr_fac)
        
        # compute initial KKT conditions
        dJdX.equals_KKT_conditions(
            x, state, adj, obj_scale=obj_fac, cnstr_scale=cnstr_fac, barrier=0.0)


        self.current_dldx.equals(dJdX)
        self.current_x.equals(x)
        state_old.equals(state)
        adj_old.equals(adj)
        
        # ----------------------- Outputing Information ----------------
        # send solution to solver
        solver_info = current_solution(
            num_iter=0, curr_primal=x.primal, curr_state=state, curr_adj=adj,
            curr_dual=x.dual)

        if isinstance(solver_info, str) and solver_info != '':
            self.info_file.write('\n' + solver_info + '\n')

        # compute convergence metrics
        if self.ineq_factory is not None:
            opt_norm0 = dJdX.primal.design.norm2
            slam_norm0 = dJdX.primal.slack.norm2
            feas_norm0 = dJdX.dual.norm2
            opt_inf0 = dJdX.primal.design.infty
            slam_inf0 = dJdX.primal.slack.infty
            feas_inf0 = dJdX.dual.infty  

        else: 
            opt_norm0 = dJdX.primal.norm2
            slam_norm0 = 0
            feas_norm0 = dJdX.dual.norm2
            opt_inf0 = dJdX.primal.infty
            slam_inf0 = 0
            feas_inf0 = dJdX.dual.infty  

        opt_tol = self.primal_tol*opt_norm0
        feas_tol = max(self.cnstr_tol*feas_norm0, 1e-6)
        self._write_header(opt_tol, feas_tol)

        # write the initial point
        obj0 = objective_value(x.primal, state)
        lag0 = obj_fac * obj0 + cnstr_fac * x0.dual.inner(dJdX.dual)
        cost0 = self.primal_factory._memory.cost
        mu0 = self.mu
        self._write_outer(0, cost0, obj0, lag0, opt_norm0, slam_norm0, feas_norm0, mu0, opt_inf0, slam_inf0, feas_inf0)
        self.hist_file.write('\n')
        # ---------------------------------------------------------------- 

        # compute the rhs vector for the predictor problem
        rhs_vec.equals(dJdX)
        rhs_vec.times(-1.)

        kkt_work.equals(x)
        kkt_work.minus(x0)
        rhs_vec.primal.plus(kkt_work.primal)
        rhs_vec.dual.minus(kkt_work.dual)

        # compute the tangent vector    # steepest descent direction when mu = 1.0;  p = -dJdX
        t.equals(0.0)
        self.hessian.linearize(
            x, state, adj,
            obj_scale=obj_fac, cnstr_scale=cnstr_fac)

        self.krylov.outer_iters = 0
        self.krylov.inner_iters = 0
        self.krylov.mu = 1.0
        self.krylov.step = 'Predictor'
        self.krylov.solve(self._mat_vec, rhs_vec, t, self.eye_precond)

        if self.symmetric is True and self.ineq_factory is not None:
            # unpeal the S^-1 layer for the slack term
            t.primal.slack.times(self.current_x.primal.slack)
        
        # normalize tangent vector
        tnorm = np.sqrt(t.inner(t) + 1.0)
        t.times(1./tnorm)
        dmu = -1./tnorm

        # START OUTER ITERATIONS
        #########################
        outer_iters = 1
        inner_iters = 0
        total_iters = 0
        corrector = False
        opt_succeed = False

        while self.mu > 0.0 and outer_iters <= self.max_iter:

            self.info_file.write(
                '==================================================\n')
            self.info_file.write(
                'Outer Homotopy iteration %i\n'%(outer_iters))
            self.info_file.write('\n')

            dJdX.equals_KKT_conditions(
                x, state, adj,
                obj_scale=obj_fac, cnstr_scale=cnstr_fac, barrier=0.0)
            opt_norm = dJdX.primal.norm2
            feas_norm = dJdX.dual.norm2
            self.info_file.write(
                'opt_norm : opt_tol = %e : %e\n'%(
                    opt_norm, opt_tol) +
                'feas_norm : feas_tol = %e : %e\n' % (
                    feas_norm, feas_tol))

            # save current solution in case we need to revert
            x_save.equals(x)
            state_save.equals(state)
            adj_save.equals(adj)
            dJdX_save.equals(dJdX)
            t_save.equals(t)
            dmu_save = dmu
            mu_save = self.mu


            # influence of mu on step size
            dmu_step = dmu * self.step
            # print 'dmu_step', dmu_step
            dmu_step = max(self.dmu_min, dmu_step)
            dmu_step = min(self.dmu_max, dmu_step)

            tent_mu = self.mu + dmu_step
            if tent_mu < 0.0:
                tent_mu = 0.0
            # -----------------------------
            max_mu_step = (tent_mu - self.mu)/dmu

            if self.mu < 1.0: 
                if self.ineq_factory is not None:
                    self.step, ind_active_s0 = self.find_step(max_mu_step, x, t)
                    t.primal.slack.base.data[ind_active_s0] = 0.0
                    # t.dual.base.data[ind_inactive_lam0] = 0.0 
                else:
                    self.step = max_mu_step
            else:
                self.step = max_mu_step

            x.equals_ax_p_by(1.0, x, self.step, t)
            self.mu += self.step*dmu

            self.info_file.write('\nmu after pred  = %.14f\n'%self.mu)

            # solve states
            if self.ineq_factory is not None:
                x.primal.design.enforce_bounds()
                if self.eq_factory is not None:
                    x.dual.ineq.base.data[x.dual.ineq.base.data > 0.0] = 0.0
                else:
                    x.dual.base.data[x.dual.base.data > 0.0] = 0.0
            else:
                x.primal.enforce_bounds()
        
            if not state.equals_primal_solution(x.primal):
                raise RuntimeError(
                    'Invalid predictor point! State-solve failed.')
            if self.factor_matrices:
                factor_linear_system(x.primal, state)

            # # compute adjoint
            adj.equals_lagrangian_adjoint(
                x, state, state_work, obj_scale=obj_fac, cnstr_scale=cnstr_fac)
            
            if self.mu < self.mu_correction:              
                
                corrector = True
                # START CORRECTOR (Newton) ITERATIONS
                #####################################
                max_newton = self.inner_maxiter
                if self.mu < EPS:
                    max_newton = self.inner_maxiter*3
                inner_iters = 0
                corrector_succeed = False
                dx_newt.equals(0.0)

                for i in xrange(max_newton):

                    self.info_file.write('\n')
                    self.info_file.write('   Inner Newton iteration %i\n'%(i+1))
                    self.info_file.write('   -------------------------------\n')

                    # compute the KKT conditions
                    dJdX.equals_KKT_conditions(
                        x, state, adj,
                        obj_scale=obj_fac, cnstr_scale=cnstr_fac, barrier=0.0)

                    # --------------------------------
                    dx_bfgs.equals(x)
                    dx_bfgs.minus(self.current_x)
                    
                    X_oldx.equals(x)
                    if self.ineq_factory is not None:
                        X_oldx.primal.design.equals(self.current_x.primal.design)
                    else:
                        X_oldx.primal.equals(self.current_x.primal)

                    dldx_bfgs.equals_KKT_conditions(
                        X_oldx, state_old, adj_old, barrier=0.0) 

                    dldx_bfgs.minus(dJdX)
                    dldx_bfgs.times(-1.0)

                    self.current_dldx.equals(dJdX)
                    state_old.equals(state)
                    adj_old.equals(adj)
                    self.current_x.equals(x)
                    # --------------------------------

                    if self.mu < EPS and inner_iters == 0:
                        opt_norm_cur = dJdX.primal.norm2
                        feas_norm_cur = dJdX.dual.norm2
                        self.inner_tol = min(opt_tol/opt_norm_cur, feas_tol/feas_norm_cur)

                    dJdX_hom.equals(dJdX)
                    dJdX_hom.times(1. - self.mu)

                    kkt_work.equals(x)
                    kkt_work.minus(x0)
                    xTx = kkt_work.primal.inner(kkt_work.primal)
                    mTm = kkt_work.dual.inner(kkt_work.dual)
                    kkt_work.times(self.mu)
                    dJdX_hom.primal.plus(kkt_work.primal)
                    dJdX_hom.dual.minus(kkt_work.dual)

                    # get convergence norms
                    if inner_iters == 0:
                        # compute optimality norms
                        hom_opt_norm0 = dJdX_hom.primal.norm2
                        hom_opt_norm = hom_opt_norm0
                        hom_opt_tol = self.inner_tol * hom_opt_norm0
                        if hom_opt_tol < opt_tol or self.mu == 0.0:
                            hom_opt_tol = opt_tol
                        # compute feasibility norms
                        hom_feas_norm0 = dJdX_hom.dual.norm2
                        hom_feas_norm = hom_feas_norm0
                        hom_feas_tol = self.inner_tol * hom_feas_norm0
                        if hom_feas_tol < feas_tol or self.mu == 0.0:
                            hom_feas_tol = feas_tol
                    else:
                        hom_opt_norm = dJdX_hom.primal.norm2
                        hom_feas_norm = dJdX_hom.dual.norm2

                    self.info_file.write(
                        '   hom_opt_norm : hom_opt_tol = %e : %e\n'%(
                            hom_opt_norm, hom_opt_tol) +
                        '   hom_feas_norm : hom_feas_tol = %e : %e\n'%(
                            hom_feas_norm, hom_feas_tol))

                    obj = objective_value(x.primal, state)
                    lag = obj_fac * obj + cnstr_fac * x.dual.inner(dJdX.dual)

                    hom = (1. - self.mu) * lag + 0.5 * self.mu * (xTx - mTm)

                    if self.ineq_factory is not None:
                        opt_norm = dJdX.primal.design.norm2
                        slam_norm = dJdX.primal.slack.norm2
                        feas_norm = dJdX.dual.norm2 
                        opt_inf = dJdX.primal.design.infty
                        slam_inf = dJdX.primal.slack.infty
                        feas_inf = dJdX.dual.infty      
                    else:
                        opt_norm = dJdX.primal.norm2
                        slam_norm = 0
                        feas_norm = dJdX.dual.norm2 
                        opt_inf = dJdX.primal.infty
                        slam_inf = 0
                        feas_inf = dJdX.dual.infty                              

                    self._write_inner(
                        outer_iters, inner_iters,
                        obj, lag, opt_norm, slam_norm, feas_norm,
                        hom, hom_opt_norm, hom_feas_norm, opt_inf, slam_inf, feas_inf)


                    if opt_norm <= opt_tol and feas_norm <= feas_tol:
                        self.info_file.write('\n  Optimization Completed!\n')
                        opt_succeed = True
                        break

                    # check convergence
                    if hom_opt_norm <= hom_opt_tol and hom_feas_norm <= hom_feas_tol:
                        self.info_file.write('\n  Corrector step converged!\n')
                        corrector_succeed = True
                        break

                    # linearize the hessian at the new point
                    self.hessian.linearize(
                        x, state, adj,
                        obj_scale=obj_fac, cnstr_scale=cnstr_fac)

                    # --------------------- Linearizing Preconditioners ------------------------                    
                    if self.svd_inequ is not None and self.mu <= self.precond_on_mu:
                        if self.ineq_factory is not None:
                            self.svd_inequ.linearize(x, state, adj, self.mu, dx_bfgs.primal.design, dldx_bfgs.primal.design)
                        else:
                            self.svd_inequ.linearize(x, state, adj, self.mu, dx_bfgs.primal, dldx_bfgs.primal)
                    
                    if self.svd_gen is not None and self.mu <= self.precond_on_mu:
                        if self.ineq_factory is not None:
                            self.svd_gen.linearize(x, state, adj, self.mu, dx_bfgs.primal.design, dldx_bfgs.primal.design)
                        else:
                            self.svd_gen.linearize(x, state, adj, self.mu, dx_bfgs.primal, dldx_bfgs.primal)                        

                    self.krylov.outer_iters = outer_iters
                    self.krylov.inner_iters = inner_iters
                    self.krylov.mu = self.mu
                    self.krylov.step = 'Corrector'

                    # define the RHS vector for the homotopy system
                    dJdX_hom.times(-1.)

                    # solve the system
                    dx.equals(0.0)

                    if self.mu <= self.precond_on_mu:
                        self.krylov.solve(self._mat_vec, dJdX_hom, dx, self.precond)
                    else:
                        self.krylov.solve(self._mat_vec, dJdX_hom, dx, self.eye_precond)

                    if self.symmetric is True and self.ineq_factory is not None: 
                        # unpeal the S^-1 layer for the slack term
                        dx.primal.slack.times(self.current_x.primal.slack)

                    # update the design
                    x.plus(dx)

                    if self.ineq_factory is not None:
                        x.primal.design.enforce_bounds()
                    else:
                        x.primal.enforce_bounds()


                    dx.equals(x)
                    dx.minus(self.current_x)
                    dx_newt.plus(dx)


                    if not state.equals_primal_solution(x.primal):
                        raise RuntimeError('Newton step failed!')
                    if self.factor_matrices:
                        factor_linear_system(x.primal, state)

                    # compute the adjoint
                    adj.equals_lagrangian_adjoint(
                        x, state, state_work,
                        obj_scale=obj_fac, cnstr_scale=cnstr_fac)

                    if self.mu < EPS:
                        solver_info = current_solution(
                            num_iter=inner_iters, curr_primal=x.primal,
                            curr_state=state, curr_adj=adj, curr_dual=x.dual)

                    # advance iter counter
                    inner_iters += 1
                    total_iters += 1

                # if we finished the corrector step at mu=0, we're done!
                if self.mu < EPS:    
                    self.info_file.write('\n>> Optimization DONE! <<\n')

                    if self.ineq_factory is not None:
                        x.primal.slack.base.data[x.primal.slack.base.data < 0.0] = 0.0
                        if self.eq_factory is not None:
                            x.dual.ineq.base.data[x.dual.ineq.base.data > 0.0] = 0.0
                        else:
                            x.dual.base.data[x.dual.base.data > 0.0] = 0.0
                    
                    # send solution to solver
                    solver_info = current_solution(
                        num_iter=outer_iters, curr_primal=x.primal,
                        curr_state=state, curr_adj=adj, curr_dual=x.dual)
                    if isinstance(solver_info, str) and solver_info != '':
                        self.info_file.write('\n' + solver_info + '\n')
                    return

            # COMPUTE NEW TANGENT VECTOR
            ############################
            # compute the KKT conditions
            dJdX.equals_KKT_conditions(
                x, state, adj,
                obj_scale=obj_fac, cnstr_scale=cnstr_fac, barrier=0.0)

            if opt_succeed is True:
                break

            if corrector_succeed is False:
                # --------------------------------
                dx_bfgs.equals(x)
                dx_bfgs.minus(self.current_x)

                X_oldx.equals(x)
                if self.ineq_factory is not None:
                    X_oldx.primal.design.equals(self.current_x.primal.design)
                else:
                    X_oldx.primal.equals(self.current_x.primal)

                dldx_bfgs.equals_KKT_conditions(
                    X_oldx, state_old, adj_old, barrier=0.0) 

                dldx_bfgs.minus(dJdX)
                dldx_bfgs.times(-1.0)

                self.current_dldx.equals(dJdX)
                state_old.equals(state)
                adj_old.equals(adj)
                self.current_x.equals(x)

            # assemble the predictor RHS
            rhs_vec.equals(dJdX)
            rhs_vec.times(-1.)

            kkt_work.equals(x)
            kkt_work.minus(x0)
            rhs_vec.primal.plus(kkt_work.primal)
            rhs_vec.dual.minus(kkt_work.dual)

            # ---------------------------------
            # for BFGS approximation of W = (1-mu)KKT + mu * I 
            dJdX_hom.equals(dJdX)
            dJdX_hom.times(1. - self.mu)

            xTx = kkt_work.primal.inner(kkt_work.primal)
            mTm = kkt_work.dual.inner(kkt_work.dual)
            kkt_work.times(self.mu)
            dJdX_hom.primal.plus(kkt_work.primal)
            dJdX_hom.dual.minus(kkt_work.dual)

            if corrector is False:
                # ------------------------------------------------
                # --------- write inner for predictor step -------
                hom_opt_norm = dJdX_hom.primal.norm2
                hom_feas_norm = dJdX_hom.dual.norm2

                obj = objective_value(x.primal, state)
                lag = obj_fac * obj + cnstr_fac * x.dual.inner(dJdX.dual)
                hom = (1. - self.mu) * lag + 0.5 * self.mu * (xTx - mTm)

                if self.ineq_factory is not None:
                    opt_norm = dJdX.primal.design.norm2
                    slam_norm = dJdX.primal.slack.norm2
                    feas_norm = dJdX.dual.norm2 
                    opt_inf = dJdX.primal.design.infty
                    slam_inf = dJdX.primal.slack.infty
                    feas_inf = dJdX.dual.infty   
                else:
                    opt_norm = dJdX.primal.norm2
                    slam_norm = 0
                    feas_norm = dJdX.dual.norm2 
                    opt_inf = dJdX.primal.infty
                    slam_inf = 0
                    feas_inf = dJdX.dual.infty                       

                self._write_inner(
                    outer_iters, 0,
                    obj, lag, opt_norm, slam_norm, feas_norm,
                    hom, hom_opt_norm, hom_feas_norm, opt_inf, slam_inf, feas_inf)
                # --------------------------------------------------

            # compute the new tangent vector and predictor step
            t.equals(0.0)
            self.hessian.linearize(
                x, state, adj,
                obj_scale=obj_fac, cnstr_scale=cnstr_fac)

            if self.svd_inequ is not None and self.mu <= self.precond_on_mu:
                if self.ineq_factory is not None:
                    self.svd_inequ.linearize(x, state, adj, self.mu, dx_bfgs.primal.design, dldx_bfgs.primal.design)
                else:
                    self.svd_inequ.linearize(x, state, adj, self.mu, dx_bfgs.primal, dldx_bfgs.primal)

            if self.svd_gen is not None and self.mu <= self.precond_on_mu:
                if self.ineq_factory is not None:
                    self.svd_gen.linearize(x, state, adj, self.mu, dx_bfgs.primal.design, dldx_bfgs.primal.design)
                else:
                    self.svd_gen.linearize(x, state, adj, self.mu, dx_bfgs.primal, dldx_bfgs.primal)    

            self.krylov.outer_iters = outer_iters 
            self.krylov.inner_iters = inner_iters
            self.krylov.mu = self.mu
            self.krylov.step = 'Predictor'

            if self.mu <= self.precond_on_mu:
                self.krylov.solve(self._mat_vec, rhs_vec, t, self.precond)
            else:
                self.krylov.solve(self._mat_vec, rhs_vec, t, self.eye_precond)

            if self.symmetric is True and self.ineq_factory is not None: 
                # unpeal the S^-1 layer for the slack term                
                t.primal.slack.times(self.current_x.primal.slack)

            # normalize the tangent vector
            tnorm = np.sqrt(t.inner(t) + 1.0)
            t.times(1./tnorm)
            dmu = -1./tnorm
            # print 'dmu:', dmu
            # compute distance to curve
            self.info_file.write('\n')
            dcurve = dx_newt.norm2
            self.info_file.write(
                'dist to curve = %e\n' % dcurve)

            # compute angle between steps
            uTv = t.inner(t_save) + (dmu * dmu_save)
            # print 'uTv : ', uTv
            angl = np.arccos(uTv)
            if np.isnan(angl):
                angl = 1e-8

            self.info_file.write(
                'angle         = %f\n' % (angl * 180. / np.pi))

            # compute deceleration factor
            dfac = max(np.sqrt(dcurve / self.nom_dcurve), angl / self.nom_angl)

            self.info_file.write(
                'dfac before cut out  = %f\n' % dfac )

            dfac = max(min(dfac, self.max_factor), self.min_factor)

            # apply the deceleration factor
            self.info_file.write('step          = %f\n' % self.step)
            self.info_file.write('factor        = %f\n' % dfac)
            self.step /= dfac
            self.info_file.write('step len      = %f\n' % self.step)

            # if factor is bad, go back to the previous point with new factor
            if dfac == self.max_factor:
                self.info_file.write(
                    'High curvature! Rejecting solution...\n')
                # revert solution
                x.equals(x_save)
                self.mu = mu_save
                t.equals(t_save)
                dmu = dmu_save
                state.equals(state_save)
                adj.equals(adj_save)
                dJdX.equals(dJdX_save)
                if self.factor_matrices:
                    factor_linear_system(x.primal, state)
            else:

                # # this step is accepted so send it to user
                if self.ineq_factory is not None:
                    x.primal.slack.base.data[x.primal.slack.base.data < 1e-6] = 1e-6
                    if self.eq_factory is not None:
                        x.dual.ineq.base.data[x.dual.ineq.base.data > 0.0] = 0.0
                    else:
                        x.dual.base.data[x.dual.base.data > 0.0] = 0.0

                solver_info = current_solution(
                    num_iter=outer_iters, curr_primal=x.primal,
                    curr_state=state, curr_adj=adj, curr_dual=x.dual)
                if isinstance(solver_info, str) and solver_info != '':
                    self.info_file.write('\n' + solver_info + '\n')

            # advance iteration counter
            outer_iters += 1
            self.info_file.write('\n')
            self.hist_file.write('\n')

# imports here to prevent circular errors
import numpy as np
from kona.options import BadKonaOption, get_opt
from kona.linalg.common import current_solution, factor_linear_system, objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import ReducedKKTMatrix
from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.hessian import LagrangianHessian
from kona.linalg.matrices.preconds import SvdInequ
from kona.linalg.matrices.preconds import SvdGen