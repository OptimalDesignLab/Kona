import numpy as np
import unittest
import pprint
import os, pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from kona.examples.sphere_constrained import SphereConstrained
import time, timeit
import pdb
from pyoptsparse import Optimization, OPT
from kona.linalg.matrices.hessian import LimitedMemoryBFGS



parser = argparse.ArgumentParser()
parser.add_argument("--output", help='Output directory', type=str, default='./temp/')
parser.add_argument("--task", help='what to do', choices=['opt','post'], default='opt')
parser.add_argument("--num_case", type=int, default=50)
args = parser.parse_args()
 
outdir = args.output
task = args.task
num_case = args.num_case

if not os.path.isdir(outdir):
    os.mkdir(outdir)


# -------------------
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
# -------------------

optns = {
    'max_iter' : 500, 
    'opt_tol' : 1e-7,
    'feas_tol' : 1e-7,        
    'info_file' : outdir + 'kona_info.dat',
    'hist_file' : outdir + 'kona_hist.dat',

    'homotopy' : {
        'inner_tol' : 0.1,
        'inner_maxiter' : 2,
        'init_step' : 0.05,
        'nominal_dist' : 1.0,
        'nominal_angle' : 10.0*np.pi/180.,
        'max_factor' : 30.0,                  
        'min_factor' : 0.001,                   
        'dmu_max' : -0.0005,       
        'dmu_min' : -0.9,      
    }, 

    'rsnk' : {
        'precond'       : None,           
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
        'krylov_file'   : outdir + 'kona_krylov.dat',
        'subspace_size' : 20,                                    
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
        'out_file'       : outdir + 'kona_verify.dat',
    },
}

###########################################################

if task == 'opt':
    sps = np.random.normal(loc=0, scale=1, size=(num_case,3))    
    kona_sols = np.zeros((num_case,3))

    for k in range(num_case):
        init_x = sps[k,:]

        # algorithm = kona.algorithms.Verifier
        solver = SphereConstrained(init_x, ineq=True)

        algorithm = kona.algorithms.PredictorCorrectorCnstrCond
        optimizer = kona.Optimizer(solver, algorithm, optns)
        optimizer.solve()

        kona_obj = solver.eval_obj(solver.curr_design, [])
        kona_x = solver.curr_design

        kona_sols[k,:] = kona_x

    # --------- Save Initial Points and Solutions -----------
    file_ =  outdir + str(num_case) + '.dat'     
    A_file = open(file_, 'w')
    pickle.dump([sps, kona_sols], A_file)
    A_file.close()

# ------- POST ---------
if task == 'post':
    file_ =  outdir + str(num_case) + '.dat'   
    A_file = open(file_, 'r')
    a = pickle.load(A_file)
    A_file.close()

    sps = a[0]
    kona_sols = a[1]

true_x = -1*np.ones((num_case, 3))
error_norm = np.linalg.norm(true_x - kona_sols, np.inf)
print 'error norm between Kona_x and True_x : ', error_norm


#########  Plot 3D Scatter Points -- a:  ##########

axis_fs = 12  # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 11 # axis labels' font size
# ax.set_axisbelow(True) # grid lines are plotted below


# fig = plt.figure()5
fig = plt.figure(figsize=(4.3,4), facecolor=None)
ax = fig.add_subplot(211, projection='3d')

# fig = plt.figure(figsize=plt.figaspect(0.5)*1.5) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
# ax = fig.gca(projection='3d')

# ax = Axes3D(fig)
# ax = fig.gca(projection='3d')

# -------- Sphere ---------
# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))*np.sqrt(3)
y = np.outer(np.sin(u), np.sin(v))*np.sqrt(3)
z = np.outer(np.ones(np.size(u)), np.cos(v))*np.sqrt(3)

# Plot the surface
alpha = 0.1
sphere1 = ax.plot_surface(x, y, z, color='b')
sphere1.set_facecolor((0, 0, 1, alpha))
# ax.set_aspect('equal', 'box')
# -----------------------------------------------
ax.scatter(sps[:,0], sps[:,1], sps[:,2], c='b', marker='o', alpha=0.5)
# ax.scatter(kona_sols[:,0], kona_sols[:,1], kona_sols[:,2], c='b', marker='o', alpha=0.5)
ax.scatter(-1, -1, -1, c='r', marker='*', s=100.0)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


ax.set_position([0.1, 0.1, 0.80, 0.83])       
plt.tick_params(labelsize=axis_fs)
# rect = ax.patch # a Rectangle instance
# #rect.set_facecolor('white')
# #rect.set_ls('dashed')
# rect.set_linewidth(axis_lw)
# rect.set_edgecolor('k')



# ------------- Plot Formats ---------------

plt.tick_params(labelsize=axis_fs)

# ax.xaxis.tick_bottom() # use ticks on bottom only
# ax.set_xticks(np.arange(min(np.ceil(min(sps[:,0])), -2), max(np.floor(max(sps[:,0])), 2), 1.0), minor=False)
# ax.set_yticks(np.arange(min(np.ceil(min(sps[:,1])), -2), max(np.floor(max(sps[:,1])), 2), 1.0), minor=False)
# ax.set_zticks(np.arange(min(np.ceil(min(sps[:,2])), -2), max(np.floor(max(sps[:,2])), 2), 1.0), minor=False)

# axisEqual3D(ax)
# ax.auto_scale_xyz([-2, 2.1], [-2, 2.1], [-2, 2.1])
ax.set_xticks(np.arange(-2, 2.1, 1.0), minor=False)
ax.set_yticks(np.arange(-2, 2.1, 1.0), minor=False)
ax.set_zticks(np.arange(-2, 2.1, 1.0), minor=False)
# scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']); ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)


ax.view_init(elev=18., azim=-60)
#plt.zticks(np.arange(-2, 2.0, 1.0))

# ax.yaxis.tick_left()
for line in ax.xaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for line in ax.yaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for line in ax.zaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.zaxis.get_ticklabels():
    label.set_fontsize(label_fs)


plt.show()

#############################################

if __name__ == "__main__":
    unittest.main()