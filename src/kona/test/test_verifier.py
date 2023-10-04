import unittest
from kona import Optimizer
from kona.algorithms import Verifier
from kona.examples import Sellar

class VerifierTestCase(unittest.TestCase):

    def test_sellar(self):
        '''Verifier test on Sellar problem'''
        solver = Sellar()

        optns = {
            'verify' : {
                'primal_vec'     : True,
                'state_vec'      : True,
                'dual_vec_eq'    : False,
                'dual_vec_in'    : True,
                'gradients'      : True,
                'pde_jac'        : True,
                'cnstr_jac_eq'   : False,
                'cnstr_jac_in'   : True,
                'red_grad'       : True,
                'lin_solve'      : True,
                'out_file'       : 'kona_verify.dat',
            },
        }

        algorithm = Verifier
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()

        self.assertTrue('Output inspected by hand...')

if __name__ == "__main__":
    unittest.main()
