import unittest

import numpy

from kona.linalg.solvers.krylov import GCROT
from kona.linalg.matrices.common import IdentityMatrix
from kona.user import UserSolver
from kona.linalg.memory import KonaMemory

class GCROTSolverTestCase(unittest.TestCase):

    def setUp(self):
        N = 100
        solver = UserSolver(N+1,0,0)
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.pf.request_num_vectors(7)
        optns = {
            'max_iter' : 10,
            'max_recycle' : 10,
            'max_outer' : 100,
            'max_krylov' : 100, # this should be hit first
            'rel_tol' : 1e-15,
        }
        self.krylov = GCROT(self.pf, optns)
        self.km.allocate_memory()

        self.x = self.pf.generate()
        self.b = self.pf.generate()
        self.b.equals(0.0)
        self.A = \
            numpy.diag(-0.5*numpy.ones(N),-1) + \
            numpy.diag(0.5*numpy.ones(N),1)
        self.A[0,0] = 0.5
        self.A[N,N] = 0.5
        self.b._data.data[0] = 1.0
        self.precond = IdentityMatrix()

    def mat_vec(self, in_vec, out_vec):
        in_data = in_vec._data.data.copy()
        out_data = self.A.dot(in_data)
        out_vec._data.data[:] = out_data[:]

    def test_dummy(self):
        self.failUnless('Untested')

    # def test_solve(self):
    #     # reset the solution vector
    #     self.x.equals(0)
    #     # solve the system with FGMRES
    #     self.krylov.solve(self.mat_vec, self.b, self.x, self.precond.product)
    #     # expected result
    #
    #     expected = numpy.array([9.948635073470220e-01, 9.848408944296067e-01,
    #     9.749371262424834e-01, 9.643563880125637e-01, 9.531488162846460e-01,
    #     9.440938633905247e-01, 9.341827526660774e-01, 9.236659797138547e-01,
    #     9.142209753530299e-01, 9.032131660932338e-01, 8.912922471692889e-01,
    #     8.827788865706691e-01, 8.732241423614135e-01, 8.624670986796664e-01,
    #     8.524144275628854e-01, 8.415880828544521e-01, 8.303687669834995e-01,
    #     8.209909146162405e-01, 8.112281459048847e-01, 8.013710645424971e-01,
    #     7.920631232266903e-01, 7.817293140824966e-01, 7.712041624350714e-01,
    #     7.602091880840678e-01, 7.489128834133452e-01, 7.403219440685793e-01,
    #     7.317163897842849e-01, 7.211037015499173e-01, 7.103846208248334e-01,
    #     6.992374567388109e-01, 6.883066122696965e-01, 6.794542066533953e-01,
    #     6.702958848170549e-01, 6.591249858446123e-01, 6.481773871707830e-01,
    #     6.385011909355367e-01, 6.287640363316058e-01, 6.190156994686001e-01,
    #     6.094039599010568e-01, 5.991883730717660e-01, 5.889114185550276e-01,
    #     5.783174530912535e-01, 5.675779396809955e-01, 5.579817510561494e-01,
    #     5.483408483086956e-01, 5.369981625037362e-01, 5.258026453575868e-01,
    #     5.164793410581492e-01, 5.070749459031826e-01, 4.966244910480385e-01,
    #     4.861875324078881e-01, 4.756782110259239e-01, 4.649648594263012e-01,
    #     4.541031874143194e-01, 4.437796378001859e-01, 4.349675933770992e-01,
    #     4.262216571793729e-01, 4.163052210410086e-01, 4.058851693505542e-01,
    #     3.953916487412998e-01, 3.846708006680632e-01, 3.737123422061620e-01,
    #     3.636441036756070e-01, 3.551644495231128e-01, 3.460838864220629e-01,
    #     3.344942796416877e-01, 3.230783225281065e-01, 3.145590553729611e-01,
    #     3.065844383561354e-01, 2.966759420354665e-01, 2.860407012559121e-01,
    #     2.759837982103739e-01, 2.656457757026411e-01, 2.541216835021101e-01,
    #     2.425165394046878e-01, 2.318891063296140e-01, 2.214678423055367e-01,
    #     2.105258947496741e-01, 2.001971387520528e-01, 1.916594002189618e-01,
    #     1.833961508484063e-01, 1.737882076939439e-01, 1.636258767302349e-01,
    #     1.537177679272952e-01, 1.435459836890796e-01, 1.327114442236212e-01,
    #     1.219591719954192e-01, 1.119160166977206e-01, 1.019428369448488e-01,
    #     9.157676725549838e-02, 8.148750655629174e-02, 7.200111113019626e-02,
    #     6.220370992317099e-02, 5.136344318296445e-02, 4.000619921693772e-02,
    #     2.894249625991445e-02, 1.855152165854391e-02, 9.519717109346798e-03,
    #     3.284649320874005e-03, 5.417980612002247e-04, 0.000000000000000e+00])
    #
    #     # compare actual result to expected
    #     diff = abs(self.x._data.data - expected)
    #     diff = max(diff)
    #     self.assertTrue(diff < 1.e-12)

if __name__ == "__main__":

    unittest.main()
