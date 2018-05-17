import numpy as np
import unittest
import pprint
import os

import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
import time, timeit
import pdb
from pyoptsparse import Optimization, OPT
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.examples import Constructed_SVDA


class InequalityTestCase(unittest.TestCase):

    def setUp(self):

        num_design = 100
        self.outdir = './temp'  
        # self.outdir = './output2/' + str(num_design) + '/'
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        self.num_design = num_design
        self.num_ineq = num_design
        np.random.seed(0) 
        self.init_x = np.random.rand(num_design)  

        if num_design==100:
            self.init_s = 40
        if num_design==200:
            self.init_s = 60
        if num_design==300:
            self.init_s = 80  
        if num_design==400:
            self.init_s = 100
        if num_design==500:
            self.init_s = 120

    def kona_optimize(self, pc):

        # Optimizer
        if pc is 'svd_inequ':
            f_info = self.outdir+'/kona_svd_info.dat'
            f_hist = self.outdir+'/kona_svd_hist.dat'
            f_krylov = self.outdir+'/kona_svd_krylov.dat'
            f_verify = self.outdir+'/kona_svd_verify.dat'
            f_optns = self.outdir+'/kona_svd_optns.dat'
            f_timing = self.outdir+'/kona_svd_timings.dat'
            krylov_sub = 20
        else:
            f_info = self.outdir+'/kona_eye_info.dat'
            f_hist = self.outdir+'/kona_eye_hist.dat'
            f_krylov = self.outdir+'/kona_eye_krylov.dat'
            f_verify = self.outdir+'/kona_eye_verify.dat'
            f_optns = self.outdir+'/kona_eye_optns.dat'
            f_timing = self.outdir+'/kona_eye_timings.dat'
            krylov_sub = 20

        optns = {
            'max_iter' : 100,
            'opt_tol' : 1e-7,
            'feas_tol' : 1e-7,        
            'info_file' : f_info,
            'hist_file' : f_hist,

            'homotopy' : {
                'init_homotopy_parameter' : 1.0, 
                'inner_tol' : 0.1,
                'inner_maxiter' : 2,
                'init_step' : self.init_s,
                'nominal_dist' : 10.0,
                'nominal_angle' : 20.0*np.pi/180.,
                'max_factor' : 50.0,                  
                'min_factor' : 0.001,                   
                'dmu_max' : -0.0005,       
                'dmu_min' : -0.9,      
                'mu_correction' : 1.0,  
                'mu_pc_on' : 1.0,      
            }, 

            'svd' : {
                'lanczos_size'    : 2, 
                'bfgs_max_stored' : 10, 
                'beta'         : 1.0, 
            }, 

            'rsnk' : {
                'precond'       : pc,      #'svd_inequ',                 
                # rsnk algorithm settings
                'dynamic_tol'   : False,
                'nu'            : 0.95,
                # reduced KKT matrix settings
                'product_fac'   : 0.001,
                'lambda'        : 0.0,
                'scale'         : 1.0,
                'grad_scale'    : 1.0,
                'feas_scale'    : 1.0,
                # FLECS solver settings
                'krylov_file'   : f_krylov,
                'subspace_size' : krylov_sub,                                    
                'check_res'     : False,
                'rel_tol'       : 1e-2,        
            },

            'verify' : {
                'primal_vec'     : True,
                'state_vec'      : False,
                'dual_vec_eq'    : False,
                'dual_vec_in'    : True,
                'gradients'      : True,
                'pde_jac'        : False,
                'cnstr_jac_eq'   : False,
                'cnstr_jac_in'   : True,
                'red_grad'       : True,
                'lin_solve'      : True,
                'out_file'       : f_verify,
            },
        }

        # pprint.pprint(optns['homotopy'])
        with open(f_optns, 'w') as file:
            pprint.pprint(optns, file)
        # algorithm = kona.algorithms.Verifier
        self.solver = Constructed_SVDA(self.num_design, self.num_ineq, self.init_x, self.outdir, f_timing)

        algorithm = kona.algorithms.PredictorCorrectorCnstrCond
        optimizer = kona.Optimizer(self.solver, algorithm, optns)
        optimizer.solve()

        self.kona_obj = self.solver.eval_obj(self.solver.curr_design, self.solver.curr_state)
        self.kona_x = self.solver.curr_design
        print 'postive dual:   ', self.solver.curr_dual[self.solver.curr_dual > 1e-5]
        print 'negative slack: ', self.solver.curr_slack[self.solver.curr_slack < -1e-5]

    def objfunc(self, xdict):
        self.iteration += 1
        self.fun_obj_counter += 1

        x = xdict['xvars']
        funcs = {}
        funcs['obj'] = 0.5 * np.dot(x.T, np.dot(self.solver.Q, x)) + np.dot(self.solver.g, x) 

        conval = np.dot(self.solver.A,x) - self.solver.b
        funcs['con'] = conval
        fail = False

        return funcs, fail

    def sens(self, xdict, funcs):

        x = xdict['xvars']
        funcsSens = {}
        funcsSens['obj'] = {'xvars': np.dot(x.T, self.solver.Q) + self.solver.g }
        funcsSens['con'] = {'xvars': self.solver.A }
        fail = False

        # ---------- recording ----------
        self.iteration += 1
        self.sens_counter += 1
        self.endTime_sn = timeit.default_timer()
        self.duration_sn = self.endTime_sn - self.startTime_sn
        self.totalTime_sn += self.duration_sn
        self.startTime_sn = self.endTime_sn

        timing = '  {0:3d}        {1:4.2f}        {2:4.2f}        {3:4.6g}     \n'.format(
            self.sens_counter, self.duration_sn, self.totalTime_sn,  funcs['obj'] )
        file = open(self.outdir+'/SNOPT_timings.dat', 'a')
        file.write(timing)
        file.close()


        return funcsSens, fail

    def optimize(self, optName, optOptions={}, storeHistory=False):

        # Optimization Object
        optProb = Optimization('SVD_Construct Problem', self.objfunc)

        # Design Variables
        value = self.init_x
        optProb.addVarGroup('xvars', self.num_design, value=value)

        # Constraints
        lower = np.zeros(self.num_ineq)
        upper = [None]*self.num_ineq
        optProb.addConGroup('con', self.num_ineq, lower = lower, upper = upper)

        # Objective
        optProb.addObj('obj')

        # Check optimization problem:
        # print(optProb)

        # Optimizer
        try:
            opt = OPT(optName, options=optOptions)
        except:
            raise unittest.SkipTest('Optimizer not available:', optName)

        # Solution
        if storeHistory:
            histFileName = '%s_svdConstruct.hst' % (optName.lower())
        else:
            histFileName = None

        sol = opt(optProb, sens=self.sens, storeHistory=histFileName)  

        
        # Check Solution
        self.pyopt_obj = sol.objectives['obj'].value
        self.pyopt_x = np.array(map(lambda x:  sol.variables['xvars'][x].value, xrange(self.num_design)))
        


    def test_snopt_kona(self):

        # ------ Kona Opt --------
        # self.kona_optimize(None)

        self.kona_optimize('svd_inequ')

        # ------ SNOPT Opt -------

        self.iteration = 0
        self.fun_obj_counter = 0
        self.sens_counter = 0

        self.startTime_sn = timeit.default_timer()  
        self.totalTime_sn = 0
        self.endTime_sn = 0
        file = open(self.outdir+'/SNOPT_timings.dat', 'w')
        file.write('# SNOPT iteration timing history\n')
        titles = '# {0:s}    {1:s}    {2:s}    {3:s}  \n'.format(
            'Iter', 'Time (s)', 'Total Time (s)', 'Objective')
        file.write(titles)
        file.close()

        optOptions = {'Print file':self.outdir + '/SNOPT_print.out',
                      'Summary file':self.outdir + '/SNOPT_summary.out',
                      'Problem Type':'Minimize',
                      }

        self.optimize('snopt', optOptions)
        
        # ----------------------------------------

        diff = max( abs( (self.kona_x - self.pyopt_x)/np.linalg.norm(self.pyopt_x) ) )


        err_diff = 'Kona and SNOPT solution X maximum relative difference, ' + str(diff)
        kona_obj = 'Kona objective value at the solution, ' + str(self.kona_obj)
        pyopt_obj = 'SNOPT objective value at the solution, ' + str(self.pyopt_obj)
        pos_dual = 'Positive dual, ' + str(self.solver.curr_dual[self.solver.curr_dual > 1e-5])
        neg_slack = 'Negative Slack, ' + str(self.solver.curr_slack[self.solver.curr_slack < -1e-5])

        with open(self.outdir+'/kona_optns.txt', 'a') as file:
            pprint.pprint(err_diff, file)
            pprint.pprint(kona_obj, file)
            pprint.pprint(pyopt_obj, file)
            pprint.pprint(pos_dual, file)
            pprint.pprint(neg_slack, file)


        print err_diff

        print 'kona_obj %f, '%(self.kona_obj)
        print 'pyopt_obj %f, '%(self.pyopt_obj)


if __name__ == "__main__":
    unittest.main()

