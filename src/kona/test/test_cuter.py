import numpy as np
import unittest, timeit
import pprint
import argparse, os, pdb
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from kona.examples import KONA_CUTER

"""
Problem Solution
BT1  -1.0
BT2  0.032568200
BT3  4.09301056 
BT4  3.28903771 
BT6  0.277044924
CAMEL6  -1.031628

OK0  : not OK, caused by other issue but not Kona
OK1  : solved by Kona (opt/feas low value), but accuracy not checked,
       as the correct solution for this CUTEr problem is not listed 
       http://www.cuter.rl.ac.uk/Problems/mastsif.shtml
OK2  : solved by Kona, the objective function agrees with 
       the solution objective on the problem page

                                        d0 t d         noPC   PC
ACOPR14   Equ and Ineq   38 0 28 144                              
AUG2D     Equ and Ineq   220 0 100 200                 Yes   No
AVGASA            Ineq            
BDRY2                              

BT1       Equ and Ineq                    1 
BT2       Equ and Ineq   3 0 1 2       |          5              
BT3       Equ and Ineq   5 0 3 6       |  0.05 10 10             
"""

parser = argparse.ArgumentParser()
parser.add_argument("--name", help='Cuter problem name', type=str, default='BT1')
parser.add_argument("--v", help='Number of variables', type=int, default=0)
parser.add_argument("--iniST", help='init step', type=float, default=0.05)
parser.add_argument("--nomDist", help='nominal dist', type=float, default=1.0)
parser.add_argument("--nomAngle", help='nominal angle', type=float, default=5.0)
parser.add_argument("--output", help='Ouput Directory', type=str, default='./temp2')
parser.add_argument("--precond", help='Preconditioner', type=str, default='Eye')
args = parser.parse_args()

prob_name = args.name
pc_name = args.precond    
V = args.v

solver = KONA_CUTER(prob_name, V)


if pc_name == 'Eye': 
    outdir = args.output + '/' + args.name 
    pc = None 
else: 
    outdir = args.output + '/' + args.name  + '_PC'

    if solver.num_eq == 0 and solver.num_ineq == 0:
        print 'Unconstrained Case, Not Considered In the Algorithm, Try another problem..'
        exit()

    elif solver.num_ineq == 0:
        print 'num_ineq = 0, equality only case, not considered yet'
        exit() 

    elif solver.num_eq == 0:
        print 'num_eq = 0, Inequality only case'
        pc = 'svd_pc_cmu'      

    else: 
        print 'Contains both equality and inequality constraints '
        pc = 'svd_pc5'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

# true_obj = 0.277044924     

# Optimizer
optns = {
    'max_iter' : 300,
    'opt_tol' : 1e-7,
    'feas_tol' : 1e-7, 
    'info_file' : outdir + '/kona_info.dat',
    'hist_file' : outdir + '/kona_hist.dat',

    'homotopy' : {
        'inner_tol' : 0.1,
        'inner_maxiter' : 2,
        'init_step' : args.iniST,
        'nominal_dist' : args.nomDist,
        'nominal_angle' : args.nomAngle*np.pi/180.,
        'max_factor' : 30.0,                  
        'min_factor' : 0.001,                   
        'dmu_max' : -0.0005,       
        'dmu_min' : -0.9,                     
    }, 

    'svd' : {
        'lanczos_size'    : 5, 
        'bfgs_max_stored' : 10, 
        'beta'         : 1.0, 
        'mu_min'       : 1e-2,
    }, 

    'rsnk' : {
        'precond'       : pc,     
        # krylov solver settings
        'krylov_file'   : outdir + '/kona_krylov.dat',
        'subspace_size' : 20,
        'check_res'     : True,
        'rel_tol'       : 1e-2,
    },
}

startTime = timeit.default_timer()
        
algorithm = PredictorCorrectorCnstrCond
optimizer = Optimizer(solver, algorithm, optns)
optimizer.solve()

# ---------- Book-keeping Options and Results -----------
duration = timeit.default_timer() - startTime
solution = solver.eval_obj(solver.curr_design, solver.curr_state)


f_optns = outdir + '/kona_optns.dat'
print 'solution : ', solution
print 'Time Elapse: ', duration

kona_obj = 'Kona objective value at its solution, ' + str(solution)
kona_time = 'Kona runtime, ' + str(duration)
cuter_dimension = 'num_design, num_state, num_eq, num_ineq : ' + \
        str(solver.num_design) + '  '  + \
        str(solver.num_state) + '  '  + \
        str(solver.num_eq)  + '  '  + \
        str(solver.num_ineq) \
       

with open(f_optns, 'a') as file:
    pprint.pprint(optns, file)
    pprint.pprint(kona_obj, file)
    pprint.pprint(kona_time, file)
    pprint.pprint(cuter_dimension, file)



""" 
Problem Classification: 

findProblems(objective=None, constraints=None, regular=None, 
    degree=None, origin=None, internal=None, 
    n=None, userN=None, m=None, userM=None)
    
Code has the following format
  ``OCRr-GI-N-M``

*O* (single letter) - type of objective

* ``N`` .. no objective function defined
* ``C`` .. constant objective function
* ``L`` .. linear objective function
* ``Q`` .. quadratic objective function
* ``S`` .. objective function is a sum of squares
* ``O`` .. none of the above

*C* (single letter) - type of constraints

* ``U`` .. unconstrained
* ``X`` .. equality constraints on variables
* ``B`` .. bounds on variables
* ``N`` .. constraints represent the adjacency matrix of a (linear) network
* ``L`` .. linear constraints
* ``Q`` .. quadratic constraints
* ``O`` .. more general than any of the above
  
*R* (single letter) - problem regularity

* ``R`` .. regular - first and second derivatives exist and are continuous
* ``I`` .. irregular problem

*r* (integer) - degree of the highest derivatives provided analytically 
    within the problem description, can be 0, 1, or 2

*G* (single letter) - origin of the problem

* ``A`` .. academic (created for testing algorithms)
* ``M`` .. modelling exercise (actual value not used in practical application)
* ``R`` .. real-world problem

*I* (single letter) - problem contains explicit internal variables

* ``Y`` .. yes
* ``N`` .. no

*N* (integer or ``V``) - number of variables, ``V`` = can be set by user

*M* (integer or ``V``) - number of constraints, ``V`` = can be set by user

-----------------------------------------------------------------------

1) 

import cutermgr


>>> cutermgr.findProblems('Q','L',True, degree=[2,2],origin='A', n=[1,500],m=[1,500]) 

['GOULDQP1', 'AUG3DCQP', 'GENHS28', 'AUG2DCQP', 'HS268', 
'CVXQP3', 'HS76I', 'QPNBAND', 'GMNCASE4', 'GMNCASE1', 
'NCVXQP3', 'NCVXQP2', 'PORTSNQP', 'AUG3DQP', 'HS44NEW', 
'BLOCKQP3', 'BLOCKQP4', 'BLOCKQP5', 'LISWET12', 'LISWET11', 
'LISWET10', 'STNQP2', 'AUG3DC', 'HS51', 'HS53', 
'NCVXQP9', 'NCVXQP8', 'NCVXQP1', 'NCVXQP7', 'NCVXQP5', 
'NCVXQP4', 'NASH', 'HS52', 'STCQP2', 'YAO', 
'LISWET1', 'TAME', 'CVXQP2', 'LISWET9', 'LISWET6', 
'LISWET5', 'AVGASA', 'RDW2D51F', 'CVXQP1', 'LISWET3', 
'HS44', 'PORTSQP', 'POWELL20', 'BLOCKQP1', 'BLOCKQP2', 
'STNQP1', 'HS118', 'AUG2D', 'DTOC3', 'DEGTRIDL', 
'BIGGSC4', 'ALLINQP', 'HS35MOD', 'LOTSCHD', 'RDW2D52F', 
'RDW2D52B', 'RDW2D52U', 'AUG3D', 'AUG2DQP', 'DEGENQP', 
'NCVXQP6', 'LISWET8', 'LISWET7', 'LISWET4', 'LISWET2', 
'STCQP1', 'ZECEVIC2', 'TWOD', 'MOSARQP2', 'MOSARQP1', 
'BDRY2', 'AUG2DC', 'HS76', 'HS35I', 'QPBAND', 
'DEGENQPC', 'SOSQP1', 'SOSQP2', 'HATFLDH', 'FERRISDC', 
'S268', 'AVGASB', 'HS21', 'HS35', 'RDW2D51U']

"""
