import unittest
import os, sys
lib_path = os.path.abspath(os.path.join('..','src'))
sys.path.append(lib_path)

import numpy
from usertemplate import UserTemplate
from usermemory import UserMemory
from vectors import DesignVector

# define a new "solver" class
class UserObj(UserTemplate):
    pass
    
# initialize user object
solver = UserObj(2, 0, 0)
# initialize Kona user memory
memory = UserMemory(solver, 3, 0, 0)
    
# unit tests begin here
class test_designVectors(unittest.TestCase):
        
    def test_assignment(self):
        # assign design vector
        D1 = DesignVector(memory)
        D2 = DesignVector(memory)
        D3 = DesignVector(memory)
        # check vector assignment flags
        self.assertTrue(
            all(item == True for item in memory.vecAssigned[DesignVector])
            )
        del D1, D2, D3
        
    def test_equalScalar(self):
        # assign design vector
        D1 = DesignVector(memory)
        # set a vector to ones
        D1.Equals(1.0)
        # get the vector data
        D1data = solver.kona_storage[0][D1.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 1.0 for item in D1data)
            )
        del D1
    
    def test_equalsVector(self):
        # assign design vector
        D1 = DesignVector(memory)
        D2 = DesignVector(memory)
        # set D1 vector to ones
        D1.Equals(1.0)
        # set D2 to D1
        D2.Equals(D1)
        # # get the vector data
        D2data = solver.kona_storage[0][D2.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 1.0 for item in D2data)
        )
        del D1, D2
    
    def test_add(self):
        # assign design vector
        D1 = DesignVector(memory)
        D2 = DesignVector(memory)
        # set D1 vector to ones
        D1.Equals(1.0)
        # set D2 to D1
        D2.Equals(D1)
        # Add D1 to D2
        D2 += D1
        # # get the vector data
        D2data = solver.kona_storage[0][D2.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 2.0 for item in D2data)
        )
        del D1, D2
        
    def test_subtract(self):
        # assign design vector
        D1 = DesignVector(memory)
        D2 = DesignVector(memory)
        # set D1 vector to ones
        D1.Equals(1.0)
        # set D2 to D1
        D2.Equals(D1)
        # Subtract D1 from D2
        D2 -= D1
        # get the vector data
        D2data = solver.kona_storage[0][D2.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 0.0 for item in D2data)
        )
        del D1, D2
        
    def test_multiply(self):
        # assign design vector
        D1 = DesignVector(memory)
        # set D1 vector to ones
        D1.Equals(1.0)
        # multiply it by 2.0
        D1 *= 2.0
        # get the vector data
        D1data = solver.kona_storage[0][D1.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 2.0 for item in D1data)
        )
        del D1
        
    def test_divide(self):
        # assign design vector
        D1 = DesignVector(memory)
        # set D1 vector to ones
        D1.Equals(1.0)
        # divide it by 2.0
        D1 /= 2.0
        # get the vector data
        D1data = solver.kona_storage[0][D1.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 0.5 for item in D1data)
        )
        del D1
        
    def test_AXPlusBY(self):
        # assign design vector
        D1 = DesignVector(memory)
        D2 = DesignVector(memory)
        D3 = DesignVector(memory)
        # set D1 and D2 to ones
        D1.Equals(1.0)
        D2.Equals(1.0)
        # Set D3 to 2*D1 + 3*D2
        D3.EqualsAXPlusBY(2.0, D1, 3.0, D2)
        # get the vector data
        D3data = solver.kona_storage[0][D3.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 5.0 for item in D3data)
        )
        del D1, D2, D3
        
    def test_Norm2(self):
        # assign design vector
        D1 = DesignVector(memory)
        # set D1 vector to ones
        D1.Equals(1.0)
        # check norm against expected value
        self.assertTrue(abs(D1.Norm2() - numpy.sqrt(1.**2 + 1.**2)) <= 1.e-5)
        memory._resetAssignments()
        del D1
        
    def test_deletion(self):
        # assign design vector
        D1 = DesignVector(memory)
        D2 = DesignVector(memory)
        D3 = DesignVector(memory)
        # delete all three vectors
        del D1
        del D2
        del D3
        # check vector assignment flags
        self.assertTrue(all(item == False for item in memory.vecAssigned[DesignVector]))

if __name__ == '__main__':
    unittest.main()