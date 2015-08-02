import os
import unittest

import kona

class VerifierTestCase(unittest.TestCase):

    def test_Constrained2x2(self):
        solver = kona.examples.Constrained2x2()
        optns = {
            'verify' : {
                'primal_vec'    : True,
                'state_vec'     : True,
                'dual_vec'      : True,
                'gradients'     : True,
                'pde_jac'       : True,
                'cnstr_jac'     : True,
                'red_grad'      : True,
                'lin_solve'     : True,
                'out_file'      : 'kona_verify.dat',
            }
        }
        optimizer = kona.Optimizer(solver, kona.algorithms.Verifier, optns)
        optimizer.solve()

        filename = os.path.realpath(__file__)
        dirname = os.path.dirname(filename)
        expected = open(dirname + '/expected_verification_output.dat', 'r')
        tested = open('kona_verify.dat', 'r')

        line1, line2 = expected.readline(), tested.readline()

        files_are_same = True
        while line1 and line2:
            if line1 != line2:
                files_are_same = False

        expected.close()
        tested.close()

        self.failUnless(files_are_same)

if __name__ == "__main__":
    unittest.main()
