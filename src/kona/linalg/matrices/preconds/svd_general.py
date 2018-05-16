import numpy as np 
import scipy as sp
import pdb
from kona.options import get_opt
from kona.linalg.matrices.preconds import LowRankSVD
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import LagrangianHessian
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
import scipy.sparse as sps

class SvdGen(BaseHessian):
    """
    On top of svd_pc4.py 
    Adding Equality Constraints into the PC
    use 1st type of system reduction with mu, equations as written in the paper
    Problem type: equality and inequality constrained
    """

    def __init__(self, vector_factories, optns={}):    

        super(SvdGen, self).__init__(vector_factories, None)
        
        svd_optns = {'lanczos_size': get_opt(optns, 2, 'lanczos_size')}  
        bfgs_optns = {'max_stored': get_opt(optns, 10, 'bfgs_max_stored')}
        self.mu_exact = get_opt(optns, 1.0, 'mu_exact')
        self.sig_exact = get_opt(optns, 1.0, 'sig_exact')
        self.beta = get_opt(optns, 1.0, 'beta')
        self.mu_min = get_opt(optns, 1e-3, 'mu_min')

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
        
        self.dual_work2.equals(0.0)
        self.dual_work2.ineq.base.data = self.sig_aug * self.dual_work1.ineq.base.data 
        self.dual_work2.eq.base.data = self.sig_aug_eq * self.dual_work1.eq.base.data 

        if self.mu < self.mu_exact:
            self.Ag.T.product(self.dual_work2, out_vec)
        else:
            self.Ag.T.approx.product(self.dual_work2, out_vec)


    def linearize(self, X, state, adjoint, mu, dx_bfgs, dldx_bfgs):   # dLdX_homo, dLdX_homo_oldual, inner_iters  

        assert isinstance(X.primal, CompositePrimalVector), \
            "SVDPC() linearize >> X.primal must be of CompositePrimalVector type!"
        assert isinstance(X.dual, CompositeDualVector),  \
            "svd_pc5() linearize >> X.dual must be of CompositeDualVector type!"

        if not self._allocated:
            self.design_work0 = self.primal_factory.generate()
            self.design_work = self.primal_factory.generate()
            self.design_work2 = self.primal_factory.generate()
            self.design_work3 = self.primal_factory.generate()
            self.design_work4 = self.primal_factory.generate()

            self.inequ_work = self.ineq_factory.generate()
            self.eq_work2 = self.eq_factory.generate()
            self.eq_work3 = self.eq_factory.generate()

            self.dual_work1 = self._generate_dual()
            self.dual_work2 = self._generate_dual()

            self._allocated = True

        # ------------ Extract Data ------------
        self.at_design = X.primal.design
        self.at_slack = X.primal.slack
        self.at_dual_eq = X.dual.eq
        self.at_dual_ineq = X.dual.ineq
        self.mu = mu

        self.state = state

        self.num_design = len(X.primal.design.base.data)
        self.num_eq = len(X.dual.eq.base.data)
        self.num_ineq = len(X.dual.ineq.base.data)

        # self.at_design_data = X.primal.design.base.data
        self.at_slack_data = X.primal.slack.base.data
        if self.eq_factory is not None:
            self.at_dual_eq_data = X.dual.eq.base.data
            self.at_dual_ineq_data = X.dual.ineq.base.data
        else:
            self.at_dual_ineq_data = X.dual.base.data

        # -------- Theories are in the paper, see the manuscript, note the minus sign ---------- 
        # # Unsymmetric KKT matrix
        self.Lam_mu = (1.0-self.mu)*self.at_dual_ineq_data - self.mu*np.ones(self.num_ineq)
        self.S_mu = (1.0-self.mu)*self.at_slack_data
        self.C_mu = self.mu * self.Lam_mu - (1 - self.mu) * self.S_mu 

        self.sig_aug = 1/self.C_mu * self.Lam_mu

        # -------- Add the equality part ----------
        self.mu_limit = max(self.mu, self.mu_min)
        self.sig_aug_eq = 1/self.mu_limit * np.ones(self.num_eq)

        # ---------- Linearize ----------
        self.Ag.linearize(X.primal.design, state)

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

            self.design_work.equals(0.0)
            self.W_hat.solve(self.svd_AT_Sig_A_mu.U[j], self.design_work)
            self.Winv_U[:, j] = self.design_work.base.data

        self.asa_U = (1-self.mu) * self.asa_U
        self.asa_V = (1-self.mu) * self.asa_V

        self.Gamma_Nstar = np.dot(self.asa_S, self.asa_V.transpose()) 
        self.svd_ASA = np.dot(self.asa_U, np.dot(self.asa_S, self.asa_V.transpose()))

        beta_I = self.beta*np.ones(self.num_design) 
        self.W_mu = (1-self.mu)*beta_I + self.mu*np.ones(self.num_design)

        self.LHS = np.diag(self.W_mu) + self.svd_ASA   


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
        u_h = rhs_vec.dual.eq.base.data 
        u_g = rhs_vec.dual.ineq.base.data        

        # solve for v_x     # Unsymmetric
        self.inequ_work.base.data = ( (1-self.mu) * u_s - self.Lam_mu * u_g) / self.C_mu

        # Write the Total Constraint Jacobian for Inequalty, Equality separately. 
        if self.mu < self.mu_exact:
            self.Ag.T.product_INEQ(self.inequ_work, self.design_work)
        else:
            self.Ag.T.approx.product_INEQ(self.inequ_work, self.design_work)

        self.design_work.times(1.0-self.mu) 

        rhs_vx = u_x - self.design_work.base.data


        # 3rd block matrix * vectors
        self.eq_work2.base.data = u_h

        self.Ag.T.product_EQ(self.eq_work2, self.design_work)

        rhs_3 = rhs_vx + 1/self.mu_limit * self.design_work.base.data
        uh_3 = u_h

        rhs_2 = self.sherman_morrison(rhs_3)
        uh_2 = -1/self.mu_limit*uh_3 

        
        self.design_work4.base.data = rhs_2

        self.Ag.product_EQ(self.design_work4, self.eq_work3)

        v_h = 1/self.mu_limit * self.eq_work3.base.data  + uh_2
        v_x = rhs_2

        # solve v_g, v_s   # Correct here
        self.design_work2.base.data = v_x
        self.inequ_work.equals(0.0)

        if self.mu < self.mu_exact:
            self.Ag.product_INEQ(self.design_work2, self.inequ_work)
        else:
            self.Ag.approx.product_INEQ(self.design_work2, self.inequ_work)

        self.inequ_work.times(1.0 - self.mu)
        rhs_ug = u_g - self.inequ_work.base.data      
                

        v_s = (-self.mu * u_s + self.S_mu * rhs_ug) / self.C_mu
        v_g = ( (1-self.mu) * u_s - self.Lam_mu * rhs_ug ) / self.C_mu

        pcd_vec.primal.design.base.data = v_x
        pcd_vec.primal.slack.base.data = v_s
        pcd_vec.dual.eq.base.data = v_h
        pcd_vec.dual.ineq.base.data = v_g
        

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
