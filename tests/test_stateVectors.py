import unittest
import os, sys
lib_path = os.path.abspath(os.path.join('..','src'))
sys.path.append(lib_path)

import numpy
from usertemplate import UserTemplate
from usermemory import UserMemory
from vectors import StateVector

# define a new "solver" class
class UserObj(UserTemplate):
    pass
    
# initialize user object
solver = UserObj(2, 2, 0)
# initialize Kona user memory
memory = UserMemory(solver, 1, 3, 0)
    
# unit tests begin here
class test_designVectors(unittest.TestCase):
        
    def test_assignment(self):
        # assign design vector
        S1 = StateVector(memory)
        S2 = StateVector(memory)
        S3 = StateVector(memory)
        # check vector assignment flags
        self.assertTrue(
            all(item == True for item in memory.vecAssigned[StateVector])
            )
        del S1, S2, S3
        
    def test_equalScalar(self):
        # assign design vector
        S1 = StateVector(memory)
        # set a vector to ones
        S1.Equals(1.0)
        # get the vector data
        S1data = solver.kona_storage[1][S1.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 1.0 for item in S1data)
            )
        del S1
    
    def test_equalsVector(self):
        # assign design vector
        S1 = StateVector(memory)
        S2 = StateVector(memory)
        # set S1 vector to ones
        S1.Equals(1.0)
        # set S2 to S1
        S2.Equals(S1)
        # # get the vector data
        S2data = solver.kona_storage[1][S2.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 1.0 for item in S2data)
        )
        del S1, S2
    
    def test_add(self):
        # assign design vector
        S1 = StateVector(memory)
        S2 = StateVector(memory)
        # set S1 vector to ones
        S1.Equals(1.0)
        # set S2 to S1
        S2.Equals(S1)
        # Add S1 to S2
        S2 += S1
        # # get the vector data
        S2data = solver.kona_storage[1][S2.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 2.0 for item in S2data)
        )
        del S1, S2
        
    def test_subtract(self):
        # assign design vector
        S1 = StateVector(memory)
        S2 = StateVector(memory)
        # set S1 vector to ones
        S1.Equals(1.0)
        # set S2 to S1
        S2.Equals(S1)
        # Subtract S1 from S2
        S2 -= S1
        # get the vector data
        S2data = solver.kona_storage[1][S2.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 0.0 for item in S2data)
        )
        del S1, S2
        
    def test_multiply(self):
        # assign design vector
        S1 = StateVector(memory)
        # set S1 vector to ones
        S1.Equals(1.0)
        # multiply it by 2.0
        S1 *= 2.0
        # get the vector data
        S1data = solver.kona_storage[1][S1.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 2.0 for item in S1data)
        )
        del S1
        
    def test_divide(self):
        # assign design vector
        S1 = StateVector(memory)
        # set S1 vector to ones
        S1.Equals(1.0)
        # divide it by 2.0
        S1 /= 2.0
        # get the vector data
        S1data = solver.kona_storage[1][S1.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 0.5 for item in S1data)
        )
        del S1
        
    def test_AXPlusBY(self):
        # assign design vector
        S1 = StateVector(memory)
        S2 = StateVector(memory)
        S3 = StateVector(memory)
        # set S1 and S2 to ones
        S1.Equals(1.0)
        S2.Equals(1.0)
        # Set S3 to 2*S1 + 3*S2
        S3.EqualsAXPlusBY(2.0, S1, 3.0, S2)
        # get the vector data
        S3data = solver.kona_storage[1][S3.GetIndex()]
        # check solver storage to confirm
        self.assertTrue(
            all(item == 5.0 for item in S3data)
        )
        del S1, S2, S3
        
    def test_Norm2(self):
        # assign design vector
        S1 = StateVector(memory)
        # set S1 vector to ones
        S1.Equals(1.0)
        # check norm against expected value
        self.assertTrue(abs(S1.Norm2() - numpy.sqrt(1.**2 + 1.**2)) <= 1.e-5)
        memory._resetAssignments()
        del S1
        
    def test_deletion(self):
        # assign design vector
        S1 = StateVector(memory)
        S2 = StateVector(memory)
        S3 = StateVector(memory)
        # delete all three vectors
        del S1
        del S2
        del S3
        # check vector assignment flags
        self.assertTrue(all(item == False for item in memory.vecAssigned[StateVector]))

if __name__ == '__main__':
    unittest.main()