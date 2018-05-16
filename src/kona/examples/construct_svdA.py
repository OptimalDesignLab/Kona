import numpy as np
from kona.user import UserSolver
import pdb
from scipy import optimize
import time, timeit

class Constructed_SVDA(UserSolver):
    """
    f = 1/2 x^T Q x + g^T x 
    s.t.  g = Ax - b >= 0
    """

    def __init__(self, numdesign, numineq, init_x, outdir, file_timing):

        super(Constructed_SVDA, self).__init__(
            num_design=numdesign, num_state=0, num_eq=0, num_ineq=numineq)

        # numdesign <= numineq, A = SVD;    if numdesign > numineq, A_T = SVD; 
        # assert numdesign <= numineq, \
        #     " numdesign <= numineq for the toy problem!"

        assert len(init_x) == numdesign, \
            " initial x size is wrong"

        self.init_x = init_x

        #--------- constructing Q and A ------------  
        A_sigma = 1./np.array(range(1, numdesign+1))**2 
        A_sigma[8:] = A_sigma[8]*np.ones(len(A_sigma[8:]))
        A_sigma = np.diag( A_sigma )    #10* 

        # np.random.seed(0)   
        A_U = np.random.randint(10, size=(numineq, numdesign))   #((numineq, numdesign))  
        A_V = np.random.randint(10, size=(numdesign, numdesign))   #((numdesign, numdesign))  

        Q_U, r_U = np.linalg.qr(A_U)
        Q_V, r_V = np.linalg.qr(A_V)

        # self.A = A_sigma

        self.A = Q_U.dot(A_sigma).dot(Q_V)    #np.eye(self.num_ineq, self.num_design)     
        self.g = np.random.rand(numdesign)    #np.zeros(numdesign)              
        self.b = np.random.rand(numineq)/10      #np.ones(numdesign)             
        
        self.outdir = outdir

        self.Q_diag = 1./np.array(range(1, numdesign+1))       
        self.Q_diag[8:] = self.Q_diag[8]*np.ones( len(self.Q_diag[8:]) )
        self.Q = np.diag( self.Q_diag )    # 10* 
        # self.Q = np.eye(numdesign) 
        # import pdb; pdb.set_trace()

        print 'Condition no. self.A: ', np.linalg.cond(self.A)
        print 'Condition no. self.Q: ', np.linalg.cond(self.Q)

        # internal optimization bookkeeping
        self.iterations = 0
        self.duration = 0.
        self.totalTime = 0.
        # self.startTime = 0.
        self.file_timing = file_timing
        self.startTime = timeit.default_timer()
        file = open(file_timing, 'w')
        file.write('# Constructed_SVDA iteration timing history\n')
        titles = '# {0:s}    {1:s}    {2:s}    {3:s}    {4:s}   {5:s}   {6:s}\n'.format(
            'Iter', 'Time (s)', 'Total Time (s)', 'Objective', 'max(abs(-S*Lam))', 'negative S', 'postive Lam' )
        file.write(titles)
        file.close()

    def eval_obj(self, at_design, at_state):

        quadra = 0.5 * at_design.dot(self.Q.dot(at_design))
        result = quadra + self.g.dot(at_design)

        return result


    def eval_dFdX(self, at_design, at_state):

        result = self.Q.dot(at_design) + self.g
        return result

    def eval_ineq_cnstr(self, at_design, at_state):

        return self.A.dot(at_design) - self.b
        
    def multiply_dCINdX(self, at_design, at_state, in_vec):

        return self.A.dot(in_vec)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):

        return self.A.transpose().dot(in_vec)

    def init_design(self):
        return self.init_x

    def init_slack(self):
        at_slack = 10*np.ones(self.num_ineq)
        # at_slack = self.eval_ineq_cnstr(self.init_x, [])
        # at_slack[ at_slack<1e-3 ] = 10.0
        return (at_slack, 0)

    def enforce_bounds(self, design_vec):
        pass

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        self.curr_design = curr_design
        self.curr_state = curr_state
        self.curr_dual = curr_ineq
        self.curr_slack = curr_slack

        # time the iteration
        self.endTime = timeit.default_timer()    
        self.duration = self.endTime - self.startTime
        self.totalTime += self.duration
        self.startTime = self.endTime

        objVal = self.eval_obj(curr_design, curr_state)
         
        slack_lamda = max(abs(curr_slack*curr_ineq))
        neg_S = sum(curr_slack < -1e-4)
        pos_Lam = sum(curr_ineq > 1e-4)

        # write timing to file
        timing = '  {0:3d}        {1:4.2f}        {2:4.2f}        {3:4.6g}       {4:4.4f}    {5:3d}   {6:3d}\n'.format(
            num_iter, self.duration, self.totalTime, objVal,   slack_lamda, neg_S, pos_Lam )
        file = open(self.file_timing, 'a')
        file.write(timing)
        file.close()


    def scipy_solution(self):

        def obj(x):
            quadra = 0.5 * np.dot(x.T, np.dot(self.Q, x))    
            result = quadra + np.dot(self.g, x)  
            return result

        def jac(x):
            return (np.dot(x.T, self.Q) + self.g)

        cons = {'type':'ineq',
                'fun':lambda x: np.dot(self.A,x) - self.b,
                'jac':lambda x: self.A}

        opt = {'disp':True, 
               'ftol':1e-6,
               'iprint' : 2, 
               'maxiter':1000}

        res_cons = optimize.minimize(obj, self.init_x, jac=jac,constraints=cons,
                                     method='SLSQP', options=opt)

        # x1, x2 = res_cons['x']
        f = res_cons['fun']

        return f, res_cons['x']
            
