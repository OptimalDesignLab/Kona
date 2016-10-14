import numpy as np
import unittest

from kona import Optimizer
from kona.algorithms import PredictorCorrectorCnstr
from kona.examples import SphereConstrained

class PredictorCorrectorCnstrTestCase(unittest.TestCase):

    def test_with_simple_constrained(self):

        init_x = [1.51, 1.52, 1.53]

        solver = SphereConstrained(init_x=init_x, ineq=False)

        optns = {
            'info_file' : 'kona_info.dat',
            'max_iter' : 100,
            'opt_tol' : 1e-5,
            'feas_tol' : 1e-5,

            'homotopy' : {
                'inner_tol' : 1e-2,
                'inner_maxiter' : 20,
                'nominal_dist' : 1.0,
                'nominal_angle' : 15.0*np.pi/180.,
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
                'krylov_file'   : 'kona_krylov.dat',
                'subspace_size' : 10,
                'rel_tol'       : 0.0095,
            },
        }

        algorithm = PredictorCorrectorCnstr
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        print solver.curr_design

        expected = -1.*np.ones(solver.num_design)
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-4)

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt

        # file = open('./kona_hist.dat', 'r')
        # data = np.loadtxt(file, usecols=(0, 3, 10))
        # outers = data[:, 0]
        # obj_dump = data[:, 1]
        # mu_dump = data[:, 2]
        #
        # mu = []
        # obj = []
        # idx = 0
        # for i in range(len(outers)):
        #     if outers[i] == solver.iters[idx]:
        #         idx += 1
        #         mu.append(mu_dump[i])
        #         obj.append(obj_dump[i])
        #         if idx > len(solver.iters) - 1:
        #             break
        #
        # x = []
        # y = []
        # z = []
        # lamb = []
        # for i in range(len(solver.design_points)):
        #     x.append(solver.design_points[i][0])
        #     y.append(solver.design_points[i][1])
        #     z.append(solver.design_points[i][2])
        #     lamb.append(solver.dual_points[i][0])
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # obj_curv = ax.plot(mu, obj, '-kx')
        # ax.set_xlim([mu[0], mu[-1]])
        # ax.set_xlabel('mu')
        # ax.set_ylabel('objective')
        # plt.savefig('obj_curv.png')
        # plt.close()
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # x_curv = ax.plot(mu, x, '-bx')
        # ax.set_xlim([mu[0], mu[-1]])
        # ax.set_xlabel('mu')
        # ax.set_ylabel('x')
        # plt.savefig('x_curv.png')
        # plt.close()
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # y_curv = ax.plot(mu, y, '-rx')
        # ax.set_xlim([mu[0], mu[-1]])
        # ax.set_xlabel('mu')
        # ax.set_ylabel('y')
        # plt.savefig('y_curv.png')
        # plt.close()
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # z_curv = ax.plot(mu, z, '-gx')
        # ax.set_xlim([mu[0], mu[-1]])
        # ax.set_xlabel('mu')
        # ax.set_ylabel('z')
        # plt.savefig('z_curv.png')
        # plt.close()
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # lamb_curv = ax.plot(mu, lamb, '-kx')
        # ax.set_xlim([mu[0], mu[-1]])
        # ax.set_xlabel('mu')
        # ax.set_ylabel('multiplier')
        # plt.savefig('lamb_curv.png')
        # plt.close()


if __name__ == "__main__":
    unittest.main()
