import numpy as np
from kona.user import UserSolver

"""
Knitro's NLP1 test problem
    minimize    1000 - x1^2 - 2*x2^2 - x3^2 - x1*x2 - x1*x3
    subject to  8*x1 + 14*x2 + 7*x3         = 56
                 (x1^2 + x2^2 + x3^2 - 25)  >= 0
                0 <= (x1, x2, x3) <= 10

    and has two local solutions:
    the point (0,0,8) with objective 936.0, and
    the point (7,0,0) with objective 951.0
"""

class NLP1(UserSolver):

    def __init__(self, init_x = [1., 1., 1.]):
        super(NLP1, self).__init__(
            num_design = 3,
            num_state = 0,
            num_eq = 1,
            num_ineq = 7)
        self.init_x = np.array(init_x)

    def eval_obj(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        return 1000 - x1**2 - 2*x2**2 - x3**2 - x1*x2 - x1*x3

    def eval_dFdX(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]

        der = np.zeros_like(at_design)
        der[0] = -2*x1 - x2 - x3
        der[1] = -4*x2 - x1
        der[2] = -2*x3 - x1
        return der

    def eval_eq_cnstr(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design.data[1]
        x3 = at_design.data[2]
        return 8*x1 + 14*x2 + 7*x3 - 56


    def eval_ineq_cnstr(self, at_design, at_state):
        x1 = at_design.data[0]
        x2 = at_design.data[1]
        x3 = at_design.data[2]

        con_ineq = np.zeros(7)
        con_ineq[0] = x1**2 + x2**2 + x3**2 - 25
        con_ineq[1] = x1 
        con_ineq[2] = x2 
        con_ineq[3] = x3
        con_ineq[4] = 10 - x1 
        con_ineq[5] = 10 - x2 
        con_ineq[6] = 10 - x3   
        return con_ineq

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        in1 = in_vec[0]
        in2 = in_vec[1]
        in3 = in_vec[2]
        return 8*in1 + 14*in2 + 7*in3

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        lam1 = in_vec

        out_vec = np.zeros(3)
        out_vec[0] = 8*lam1
        out_vec[1] = 14*lam1
        out_vec[2] = 7*lam1
        return out_vec

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        in1 = in_vec[0]
        in2 = in_vec[1]
        in3 = in_vec[2]

        out_vec = np.zeros(7)
        out_vec[0] =  2*x1*in1 + 2*x2*in2 + 2*x3*in3
        out_vec[1] = in1
        out_vec[2] = in2 
        out_vec[3] = in3
        out_vec[4] = -in1
        out_vec[5] = -in2
        out_vec[6] = -in3        
        return out_vec

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        lam2 = in_vec[0]
        lam3 = in_vec[1]
        lam4 = in_vec[2]
        lam5 = in_vec[3]
        lam6 = in_vec[4]
        lam7 = in_vec[5]
        lam8 = in_vec[6]

        out_vec = np.zeros(3)        
        out_vec[0] = 2*x1*lam2 + lam3 - lam6
        out_vec[1] = 2*x2*lam2 + lam4 - lam7
        out_vec[2] = 2*x3*lam2 + lam5 - lam8
        return out_vec

    def init_design(self):
        return self.init_x
