import numpy as np 
import scipy as sp
import pdb
from kona.options import get_opt
from kona.linalg.matrices.preconds import LowRankSVD
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import LagrangianHessian
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
import scipy.sparse as sps

class SvdInequ(BaseHessian):
    """
    A general preconditioner for inequality-only case
    use 1st type of system reduction with mu, equations as written in the paper
    test different C_min values to deal with ill-condition
    """

    def __init__(self, vector_factories, optns={}):    

        super(SvdInequ, self).__init__(vector_factories, None)

        # self.primal_factory.request_num_vectors(10)
        # self.state_factory.request_num_vectors(2)
        # if self.eq_factory is not None:
        #     self.eq_factory.request_num_vectors(3)
        # if self.ineq_factory is not None:
        #     self.ineq_factory.request_num_vectors(5)
        
        svd_optns = {'lanczos_size': get_opt(optns, 40, 'lanczos_size')}  
        bfgs_optns = {'max_stored': get_opt(optns, 10, 'bfgs_max_stored')}
        self.mu_exact = get_opt(optns, -1.0, 'mu_exact')
        self.sig_exact = get_opt(optns, 1.0, 'sig_exact')
        self.beta = get_opt(optns, 1.0, 'beta')
        self.cmin = get_opt(optns, -1e-3, 'cmin')
        self.fstopo = get_opt(optns, False, 'fstopo')

        print 'fstopo Problem ? ', self.fstopo

        self.Ag = TotalConstraintJacobian( vector_factories )
        self.svd_AT_Sig_A_mu = LowRankSVD(
            self.asa_mat_vec_mu, self.primal_factory, None, None, svd_optns)
        self.W_hat = LimitedMemoryBFGS(self.primal_factory, bfgs_optns)  
        self.W_hat.norm_init = 1.0  
        
        self._allocated = False

    def asa_mat_vec_mu(self, in_vec, out_vec):

        if self.mu < self.mu_exact:
            self.Ag.product(in_vec, self.dual_work1)
        else:
            self.Ag.approx.product(in_vec, self.dual_work1)

        # self.dual_work1.times(1.0-self.mu)  # (1-mu) considered in linearization
        
        self.dual_work2.equals(0.0)
        if self.fstopo is True:     
            if self.mu < self.sig_exact:                 
                self.dual_work2.base.data[-self.num_design:] = self.sig_aug[2*self.num_design:] * self.dual_work1.base.data[-self.num_design:]
            else:
                self.dual_work2.equals(self.dual_work1)
        else: 
            self.dual_work2.base.data = self.sig_aug * self.dual_work1.base.data 

        if self.mu < self.mu_exact:
            self.Ag.T.product(self.dual_work2, out_vec)
        else:
           self.Ag.T.approx.product(self.dual_work2, out_vec)

        # out_vec.times(1.0-self.mu)


    def linearize(self, X, state, adjoint, mu, dx_bfgs, dldx_bfgs):   # dLdX_homo, dLdX_homo_oldual, inner_iters  

        assert isinstance(X.primal, CompositePrimalVector), \
            "SVDPC() linearize >> X.primal must be of CompositePrimalVector type!"
        assert isinstance(X.dual, DualVectorINEQ),  \
            "SVDPC() linearize >> X.dual must be of DualVectorINEQ type!"

        if not self._allocated:
            self.design_work0 = self.primal_factory.generate()
            self.design_work = self.primal_factory.generate()
            self.design_work2 = self.primal_factory.generate()
            self.design_work3 = self.primal_factory.generate()

            self.slack_work = self.ineq_factory.generate()
            self.kkt_work = self._generate_kkt()

            self.dual_work1 = self.ineq_factory.generate()
            self.dual_work2 = self.ineq_factory.generate()
            self.dual_work3 = self.ineq_factory.generate()

            # approximate BFGS Hessian
            self.dldx_old = self.primal_factory.generate()
            self.dldx = self.primal_factory.generate()
            self.design_old = self.primal_factory.generate()

            self._allocated = True

        # ------------ Extract Data ------------
        self.at_design = X.primal.design
        self.at_slack = X.primal.slack
        self.at_dual_ineq = X.dual
        self.mu = mu

        self.num_design = len(X.primal.design.base.data)
        self.num_ineq = len(X.dual.base.data)

        # self.at_design_data = X.primal.design.base.data
        self.at_slack_data = X.primal.slack.base.data
        if self.eq_factory is not None:
            self.at_dual_eq_data = X.dual.eq.base.data
            self.at_dual_ineq_data = X.dual.ineq.base.data
        else:
            self.at_dual_ineq_data = X.dual.base.data

        # -------- Theories are in the paper, see the manuscript, note the minus sign ---------- 
        # # Symmetric KKT matrix
        # self.Lam_mu = (1.0-self.mu)*self.at_dual_ineq_data*self.at_slack_data - self.mu*self.at_slack_data
        # self.S_mu = (1.0-self.mu)*self.at_slack_data
        # self.C_mu_orig = self.mu * self.Lam_mu - self.S_mu ** 2

        # # Unsymmetric KKT matrix
        self.Lam_mu = (1.0-self.mu)*self.at_dual_ineq_data - self.mu*np.ones(self.num_ineq)
        self.S_mu = (1.0-self.mu)*self.at_slack_data
        self.C_mu_orig = self.mu * self.Lam_mu - (1 - self.mu) * self.S_mu 

        # self.C_mu = np.minimum(-self.cmin*np.ones(self.num_ineq), self.C_mu_orig)
        self.C_mu = self.C_mu_orig
        self.sig_aug = 1/self.C_mu * self.Lam_mu

        # ---------- Linearize ----------
        self.Ag.linearize(X.primal.design, state)

        if self.fstopo is False:
            # ---------- Hessian LBFGS approximation --------  
            dldx_bfgs.equals_ax_p_by(1.0-self.mu, dldx_bfgs, self.mu, dx_bfgs)
            self.W_hat.add_correction(dx_bfgs, dldx_bfgs)


        # ----------- svd_AsT_SigS_As_mu -------------
        self.svd_AT_Sig_A_mu.linearize()      

        self.asa_S = self.svd_AT_Sig_A_mu.S 
        self.asa_U = np.zeros((self.num_design, len( self.svd_AT_Sig_A_mu.U ) ))
        self.asa_V = np.zeros((self.num_design, len( self.svd_AT_Sig_A_mu.V ) ))
        self.Winv_U = np.zeros((self.num_design, self.asa_S.shape[0] ))

        for j in xrange(self.asa_S.shape[0]):
            self.asa_U[:, j] = self.svd_AT_Sig_A_mu.U[j].base.data
            self.asa_V[:, j] = self.svd_AT_Sig_A_mu.V[j].base.data

            if self.fstopo is False:
                self.design_work.equals(0.0)
                self.W_hat.solve(self.svd_AT_Sig_A_mu.U[j], self.design_work)
                self.Winv_U[:, j] = self.design_work.base.data

        self.asa_U = (1-self.mu) * self.asa_U
        self.asa_V = (1-self.mu) * self.asa_V

        self.Gamma_Nstar = np.dot(self.asa_S, self.asa_V.transpose()) 
        self.svd_ASA = np.dot(self.asa_U, np.dot(self.asa_S, self.asa_V.transpose()))

        beta_I = self.beta*np.ones(self.num_design) 
        self.W_mu = (1-self.mu)*beta_I + self.mu*np.ones(self.num_design)

        # if self.fstopo is True:                           
        #     self.LHS = np.diag(self.W_mu + (1.0-self.mu)**2 * (self.sig_aug[: self.num_design] + \
        #         self.sig_aug[self.num_design : 2*self.num_design ]) ) +  self.svd_ASA    
        # else:
        #     self.LHS = np.diag(self.W_mu) + self.svd_ASA   


    def sherman_morrison(self, rhs_vx):

        self.design_work.base.data = rhs_vx
        self.W_hat.solve(self.design_work, self.design_work0)

        work_1 = np.dot(self.Gamma_Nstar,  self.design_work0.base.data)

        core_mat = np.eye(self.asa_S.shape[0]) + np.dot(self.Gamma_Nstar, self.Winv_U )     
        work_2 = sp.linalg.lu_solve(sp.linalg.lu_factor(core_mat), work_1) 

        work_3 = np.dot(self.asa_U, work_2)
        self.design_work2.base.data = work_3

        self.W_hat.solve(self.design_work2, self.design_work3 )
        
        out = self.design_work0.base.data - self.design_work3.base.data

        return out

    def sherman_morrison_betaI(self, rhs_vx):

        if self.fstopo is True: 
            W_aug = self.W_mu + (1.0-self.mu)**2 * (self.sig_aug[: self.num_design] + \
                self.sig_aug[self.num_design : 2*self.num_design ])  
            W_mu_inv = np.diag(1.0/W_aug)

        else: 
            W_mu_inv = np.diag(1.0/self.W_mu)


        work_1 = np.dot(self.Gamma_Nstar,  np.dot(W_mu_inv, rhs_vx))    

        core_mat = np.eye(self.asa_S.shape[0]) + np.dot(self.Gamma_Nstar,  np.dot(W_mu_inv, self.asa_U) )     

        work_2 = sp.linalg.lu_solve(sp.linalg.lu_factor(core_mat), work_1) 

        out = np.dot(W_mu_inv, rhs_vx) - np.dot(W_mu_inv, np.dot(self.asa_U, work_2))

        return  out      


    def solve(self, rhs_vec, pcd_vec):    

        # using scaled slack version,  Lambda_aug, I'' contains Slack component
        u_x = rhs_vec.primal.design.base.data
        u_s = rhs_vec.primal.slack.base.data
        u_g = rhs_vec.dual.base.data        

        # use the block matrix expression as in the paper:
        # solve for v_x

        # # Symmetric
        # self.dual_work1.base.data = (self.S_mu * u_s - self.Lam_mu * u_g) / self.C_mu

        # # Unsymmetric
        self.dual_work1.base.data = ( (1-self.mu) * u_s - self.Lam_mu * u_g) / self.C_mu

        if self.mu < self.mu_exact:
            self.Ag.T.product(self.dual_work1, self.design_work)
        else:
            self.Ag.T.approx.product(self.dual_work1, self.design_work)

        self.design_work.times(1.0-self.mu) 

        rhs_vx = u_x - self.design_work.base.data

        # v_x = sp.linalg.lu_solve(sp.linalg.lu_factor(self.LHS), rhs_vx)         
        if self.fstopo is True: 
            v_x = self.sherman_morrison_betaI(rhs_vx)
            # v_x = sp.linalg.lu_solve(sp.linalg.lu_factor(self.LHS), rhs_vx)  
        else: 
            v_x = self.sherman_morrison(rhs_vx)

        # solve v_g, v_s
        self.design_work2.base.data = v_x

        if self.mu < self.mu_exact:
            self.Ag.product(self.design_work2, self.dual_work2)
        else:
            self.Ag.approx.product(self.design_work2, self.dual_work2)

        self.dual_work2.times(1.0 - self.mu)
        rhs_ug = u_g - self.dual_work2.base.data      
                

        v_s = (-self.mu * u_s + self.S_mu * rhs_ug) / self.C_mu
        # # Symmetric Case : 
        # v_g = ( self.S_mu * u_s - self.Lam_mu * rhs_ug ) / self.C_mu

        # # Unsymmetric Case : 
        v_g = ( (1-self.mu) * u_s - self.Lam_mu * rhs_ug ) / self.C_mu

        pcd_vec.primal.design.base.data = v_x
        pcd_vec.primal.slack.base.data = v_s
        pcd_vec.dual.base.data = v_g

        

    def _generate_kkt(self):
        prim = self.primal_factory.generate()
        slak = self.ineq_factory.generate()        
        dual = self.ineq_factory.generate()
        return ReducedKKTVector(CompositePrimalVector(prim, slak), dual)
