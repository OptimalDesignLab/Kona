import unittest

import numpy as np

import kona

from kona.linalg.solvers.util import eigen_decomp, abs_sign, calc_epsilon
from kona.linalg.solvers.util import apply_givens, generate_givens, solve_tri
from kona.linalg.solvers.util import EPS

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
        self.assertTrue(abs(out2 - 1.4901161193847656e-13) < 1e-10 )

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

if __name__ == "__main__":

    unittest.main()
