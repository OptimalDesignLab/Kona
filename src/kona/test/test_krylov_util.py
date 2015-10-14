import unittest

import numpy as np

import kona

from kona.linalg.solvers.util import eigen_decomp, abs_sign, calc_epsilon
from kona.linalg.solvers.util import apply_givens, generate_givens, solve_tri
from kona.linalg.solvers.util import secular_function, solve_trust_reduced, EPS

class KrylovUtilTestCase(unittest.TestCase):

    def test_sign(self):
        x = 2
        y = -1
        z = abs_sign(x,y)

        self.assertEqual(z,-2)

    def test_calc_epsilon(self):

        mult_by_norm = 1e-30
        eval_at_norm = 1

        out1 = calc_epsilon(eval_at_norm, mult_by_norm)

        self.assertEqual(out1, 1.0)

        mult_by_norm = 1e5
        eval_at_norm = 1e-30

        out2 = calc_epsilon(eval_at_norm, mult_by_norm)
        self.assertTrue(abs(out2 - 1.4901161193847656e-13) < 1e-10)

        mult_by_norm = 1
        eval_at_norm = 2

        out3 = calc_epsilon(eval_at_norm, mult_by_norm)
        self.assertTrue(abs(out3 - 2*EPS) < 1e-5)

    def test_eigen_decomp(self):

        A = np.diag(np.array([1,2,3,4]))

        eig_val, eig_vec = eigen_decomp(A)
        self.assertTrue(np.all(eig_val == np.array([1,2,3,4])))

        self.assertTrue(np.all(eig_vec == np.eye(4)))

    def test_apply_givens(self):

        # theta = 90
        s = 1
        c = 0
        h1 = 1
        h2 = 1

        a1, a2 = apply_givens(s, c, h1, h2)

        self.assertEquals(a1, 1)
        self.assertEquals(a2, -1)

    def test_generate_givens(self):
        dx = 0.0
        dy = 0.0
        dx, dy, s, c = generate_givens(dx, dy)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)
        self.assertEqual(s, 0.0)
        self.assertEqual(c, 1.0)

        dx = 1.0
        dy = -2.0
        dx, dy, s, c = generate_givens(dx, dy)
        self.assertTrue(abs(dx - 2*1.118033989) < 1e-5)
        self.assertEqual(dy, 0.0)
        self.assertTrue(abs(s + 0.894427191) < 1e-5)
        self.assertTrue(abs(c - 0.4472135955) < 1e-5)

        dx = 2.0
        dy = -1.0
        dx, dy, s, c = generate_givens(dx, dy)
        self.assertTrue(abs(dx - 2*1.118033989) < 1e-5)
        self.assertEqual(dy, 0.0)
        self.assertTrue(abs(s + 0.4472135955) < 1e-5)
        self.assertTrue(abs(c - 0.894427191) < 1e-5)

        dx = float('nan')
        dy = float('nan')
        dx, dy, s, c = generate_givens(dx, dy)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)
        self.assertEqual(s, 0.0)
        self.assertEqual(c, 1.0)

    def test_solve_tri(self):
        A = np.matrix([[1, 1],[0, 1]])
        b = np.array([3, 1])
        x = solve_tri(A, b, lower=False)
        self.assertEqual(x[1], 1.)
        self.assertEqual(x[0], 2.)

        AT = np.transpose(A)
        x = solve_tri(AT, b, lower=True)
        self.assertEqual(x[1], -2.)
        self.assertEqual(x[0], 3.)

        kona.linalg.solvers.util.scipy_exists = False
        x = solve_tri(A, b)
        self.assertEqual(x[1], 1.)
        self.assertEqual(x[0], 2.)

    def test_secular_function(self):

        # The eigenvalues of the following matrix are (1e-5, 0.01, 1)
        A = np.zeros((3,3))
        A[0][0] = 3.931544008059447
        A[0][1] = -4.622828930484834
        A[1][0] = A[0][1]
        A[0][2] = 1.571893108754884
        A[2][0] = A[0][2]
        A[1][1] = 5.438436601890520
        A[1][2] = -1.853920290644159
        A[2][1] = A[1][2]
        A[2][2] = 0.640029390050034
        A = np.matrix(A)
        b = np.array([-0.964888535199277,
                      -0.157613081677548,
                      -0.970592781760616])

        # test secular function with lambda = 0.0 and delta = ||A\b||; the
        # secular function should vanish, and the step y should be -A\b
        radius = 106357.56613920075
        y, fnc, dfnc = secular_function(A, b, 0.0, radius)
        self.assertFalse(abs(y[0] - 70306.51598209806) > 1e-5)
        self.assertFalse(abs(y[1] - 71705.07008271456) > 1e-5)
        self.assertFalse(abs(y[2] - 35032.96463715491) > 1e-5)
        self.assertFalse(abs(fnc) > 1e-12)
        self.assertFalse(abs(dfnc + 0.9402244082089186) > 1e-12)

        # test secular function with lambda = 0.1 and delta = 10
        lamb = 0.1
        radius = 10.0
        y, fnc, dfnc = secular_function(A, b, lamb, radius)
        self.assertFalse(abs(y[0] - 5.280791785178089) > 1e-5)
        self.assertFalse(abs(y[1] - 6.941941330303563) > 1e-5)
        self.assertFalse(abs(y[2] - 7.485592503571139) > 1e-5)
        self.assertFalse(abs(fnc - 0.01299787866032312) > 1e-12)
        self.assertFalse(abs(dfnc + 0.8585317920163272) > 1e-12)

    def test_solve_trust_reduced(self):
        # first we test with the trust radius constraint inactive

        # The eigenvalues of the following matrix are (1e-5, 0.01, 1)
        A = np.zeros((3,3))
        A[0][0] = 3.931544008059447
        A[0][1] = -4.622828930484834
        A[1][0] = A[0][1]
        A[0][2] = 1.571893108754884
        A[2][0] = A[0][2]
        A[1][1] = 5.438436601890520
        A[1][2] = -1.853920290644159
        A[2][1] = A[1][2]
        A[2][2] = 0.640029390050034
        A = np.array(A)
        b = np.array([-0.964888535199277,
                      -0.157613081677548,
                      -0.970592781760616])
        radius = 1e6
        x, lamb, pred = solve_trust_reduced(A, b, radius)
        self.assertFalse(abs(x[0] - 70306.51598209806) > 1e-5)
        self.assertFalse(abs(x[1] - 71705.07008271456) > 1e-5)
        self.assertFalse(abs(x[2] - 35032.96463715491) > 1e-5)
        self.assertFalse(abs(pred - 56571.17544243777) > 1e-5)
        self.assertFalse(abs(lamb) > EPS)

        # then test with the trust radius constraint active
        radius = 10000.
        x, lamb, pred = solve_trust_reduced(A, b, radius)
        self.assertFalse(abs(x[0] - 6592.643411099528) > 1e-3*abs(x[0]))
        self.assertFalse(abs(x[1] - 6740.041501068382) > 1e-3*abs(x[1]))
        self.assertFalse(abs(x[2] - 3333.000662760490) > 1e-3*abs(x[2]))
        self.assertFalse(abs(pred - 10147.17333545226) > 1e-3*abs(pred))
        self.assertFalse(abs(lamb - 9.635875530658215e-05) > 1e-3*lamb)

        # And finally we test for indefinite hessian; the matrix A below has the
        # same eigenvalue magnitudes as those above, but the smallest eigenvalue
        # is negative.
        A = np.zeros((3,3))
        A[0][0] = 3.931535263699851
        A[0][1] = -4.622837846534464
        A[1][0] = A[0][1]
        A[0][2] = 1.571888758188687
        A[2][0] = A[0][2]
        A[1][1] = 5.438427510779841
        A[1][2] = -1.853924726631001
        A[2][1] = A[1][2]
        A[2][2] = 0.640027225520312
        A = np.array(A)
        b = np.array([-1e-5, -1e-5, -1e-5])
        radius = 10000.
        x, lamb, pred = solve_trust_reduced(A, b, radius)
        self.assertFalse(abs(x[0] - 6612.245873748023) > 1e-6*abs(x[0]))
        self.assertFalse(abs(x[1] - 6742.073357887492) > 1e-6*abs(x[1]))
        self.assertFalse(abs(x[2] - 3289.779831837675) > 1e-6*abs(x[2]))
        self.assertFalse(abs(pred - 500.1664410153899) > 1e-6*abs(pred))
        self.assertFalse(abs(lamb - 1.000166441038206e-05) > 1e-6*lamb)

if __name__ == "__main__":

    unittest.main()
