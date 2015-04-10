import unittest

import numpy as np

from kona.linalg.solvers.krylov_util import eigenvalues, sign, CalcEpsilon, applyGivens

class KrylovUtilTestCase(unittest.TestCase): 


    def test_sign(self):
        x = 2
        y = -1
        z = sign(x,y)

        self.assertEqual(z,-2)

    def test_CalcEpsilon(self):

        mult_by_norm = 1e-30
        eval_at_norm = 1

        out1 = CalcEpsilon(eval_at_norm, mult_by_norm)

        self.assertEqual(out1, 1.0)

        mult_by_norm = 1e5
        eval_at_norm = 1e-30

        out2 = CalcEpsilon(eval_at_norm, mult_by_norm)
        self.assertTrue(abs(out2-1.4901161193847656e-13)<1e-10 )


    def test_eigenvalues(self): 

        A = np.diag(np.array([1,2,3,4]))

        eig_val, eig_vec = eigenvalues(A)
        self.assertTrue(np.all(eig_val == np.array([1,2,3,4])))
        
        self.assertTrue(np.all(eig_vec == np.eye(4)))


    def test_applyGivens(self):

        # theta = 90
        s = 1
        c = 0
        h1 = 1
        h2 = 1

        a1, a2 = applyGivens(s, c, h1, h2)
        print a1
        print a2
        print 'has some doubts'

    def test_generateGivens(self):
        pass



if __name__ == "__main__": 

    unittest.main()
