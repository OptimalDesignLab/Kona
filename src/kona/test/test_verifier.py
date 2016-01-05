import unittest
import sys

from kona import Optimizer
from kona.algorithms import Verifier
from kona.examples import Sellar

class VerifierTestCase(unittest.TestCase):

    def test_sellar(self):

        solver = Sellar()

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
                'out_file'      : sys.stdout,
            },
        }

        algorithm = Verifier
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        self.failUnless('Output inspected by hand...')

if __name__ == "__main__":
    unittest.main()
