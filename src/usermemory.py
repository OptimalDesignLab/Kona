import numpy
from common import Singleton
from usertemplate import UserTemplate
from vectors import *

class UserMemory(Singleton):
    
    def __init__(self, userObj, numDesignVec, numStateVec, numDualVec):
        # assign user object
        if not isinstance(userObj, UserTemplate):
            raise TypeError('UserMemory() >> Unknown user object.')
        else:
            self.userObj = userObj
        
        # cost tracking
        self.precondCount = 0
        
        # store how many vectors of each type we will use
        self.numDesignVec = numDesignVec
        self.numStateVec = numStateVec
        self.numDualVec = numDualVec
        
        # allocate vec assignments
        self.vecAssigned = {
            DesignVector : [False] * self.numDesignVec,
            StateVector : [False] * self.numStateVec,
            DualVector : [False] * self.numDualVec,
        }
        
        # send request to user to allocate storage for vectors
        ierr = self.userObj.alloc_memory(self.numDesignVec, 
                                         self.numStateVec, 
                                         self.numDualVec)
        if ierr != 0:
            raise RuntimeError('userObj.alloc_memory() >> ' + \
                               'Error Code: ' % ierr)
    def _resetAssignments(self):
        # THIS IS ONLY USED BY UNIT TESTS
        self.vecAssigned = {
            DesignVector : [False] * self.numDesignVec,
            StateVector : [False] * self.numStateVec,
            DualVector : [False] * self.numDualVec,
        }
        # THIS IS ONLY USED BY UNIT TESTS
        
    def CheckType(self, vec, vecType):
        # make sure we're comparing to a valid vector type
        if vecType not in self.vecAssigned.keys():
            raise TypeError('UserMemory.CheckType() >> Unknown type provided.')
            
        # if the vector is not the specified type, raise error
        if not isinstance(vec, vecType):
            raise TypeError('UserMemory.CheckType() >> Type mismatch. ' + \
                            'Vector Must be %s.' % vecType)
        
    def AssignVector(self, vec):
        # loop over the storage flags of appropriate vector type
        found = False
        for index, assigned in enumerate(self.vecAssigned[type(vec)]):
            # break loop when we reach an unassigned location
            if not assigned:
                found = True
                freeSpot = index
                break
                
        # if no unassigned spots were found, raise error
        if not found:
            raise RuntimeError('UserMemory.AssignVector() >> ' + \
                               'Cannot assign new vector. No space left.')
                               
        # otherwise flip the assignment flag and return the index
        else:
            self.vecAssigned[type(vec)][freeSpot] = True
            return freeSpot
        
    def UnassignVector(self, vec):
        # change the assignment flag for the given vector type and index
        self.vecAssigned[type(vec)][vec.GetIndex()] = False
        
    def AXPlusBY(self, a, xIndex, b, yIndex, result):
        # request the user function that matches vector type
        vecType = result.GetFlag()
        resultIndex = result.GetIndex()
        self.userObj.ax_p_by(vecType, a, xIndex, b, yIndex, resultIndex)
        
    def InnerProd(self, x, y):
        # make sure the two vectors have the same type
        self.CheckType(y, type(x))
        
        # request the user function that matches vector type
        vecType = x.GetFlag()
        xIndex = x.GetIndex()
        yIndex = y.GetIndex()
        return self.userObj.inner_prod(vecType, xIndex, yIndex)
        
    def Restrict(self, opType, vec):
        # check vector types
        self.CheckType(vec, DesignVector)
        
        # send request to user
        self.userObj.restrict_design(opType, vec.GetIndex())
        
    def ConvertVec(self, source, target):
        # check vector types
        self.CheckType(source, DualVector)
        self.CheckType(target, DesignVector)
        
        # send request to user
        self.userObj.copy_dual_to_targstate(source.GetIndex(), 
                                            target.GetIndex())
        
    def SetInitialDesign(self, initDesign):
        # check vector types
        self.CheckType(initDesign, DesignVector)
        
        # send request to user
        self.userObj.init_design(initDesign.GetIndex())
        
    def EvaluateObjective(self, atDesign, atState):
        # check vector types
        self.CheckType(atDesign, DesignVector)
        self.CheckType(atState, StateVector)
        
        # send request to user and return result
        return self.userObj.eval_obj(atDesign.GetIndex(), 
                                     atState.GetIndex())
        
    def EvaluatePDEResidual(self, atDesign, atState, PDEresidual):
        # check vector types
        self.CheckType(atDesign, DesignVector)
        self.CheckType(atState, StateVector)
        self.CheckType(PDEresidual, StateVector)
        
        # send request to user
        self.userObj.eval_residual(atDesign.GetIndex(), 
                                   atState.GetIndex(),
                                   residual.GetIndex())
        
    def EvaluateConstraints(self, atDesign, atState, constraints):
        # check vector types
        self.CheckType(atDesign, DesignVector)
        self.CheckType(atState, StateVector)
        self.CheckType(constraints, DualVector)
        
        # send request to user
        self.userObj.eval_ceq(atDesign.GetIndex(), 
                              atState.GetIndex(),
                              constraints.GetIndex())
        
    def SolvePDE(self, atDesign, result):
        # check vector types
        self.CheckType(atDesign, DesignVector)
        self.CheckType(result, StateVector)
        
        # send request to user
        cost = self.userObj.solve_system(atDesign.GetIndex(), 
                                         result.GetIndex())
                                         
        # update cost tracking and return convergence info
        self.precondCount += abs(cost)
        if cost < 0:
            return False
        else:
            return True
        
    def SolveLinearized(self, atDesign, atState, rhsVec, relTol, result):
        # check vector types
        self.CheckType(atDesign, DesignVector)
        self.CheckType(atState, StateVector)
        self.CheckType(rhsVec, StateVector)
        self.CheckType(result, StateVector)
        
        # check solution tolerance
        if relTol < 0.0:
            raise ValueError('UserMemory.SolveLinearized() >> ' + \
                             'Negative tolerance.')
                             
        # send the request to the user
        cost = self.userObj.solve_linearsys(atDesign.GetIndex(), 
                                            atState.GetIndex(),
                                            rhsVec.GetIndex(), tol,
                                            result.GetIndex())
                                            
        # update cost tracking and return convergence info
        self.precondCount += abs(cost)
        if cost < 0:
            return False
        else:
            return True
            
    def SolveAdjoint(self, atDesign, atState, rhsVec, relTol, result):
        # check vector types
        self.CheckType(atDesign, DesignVector)
        self.CheckType(atState, StateVector)
        if isinstance(rhsVec, StateVector):
            rhsIndex = rhsVec.GetIndex()
        elif rhsVec == -1:
            rhsIndex = rhsVec
        else:
            raise TypeError('UserMemory.SolveAdjoint() >> Wrong RHS type.')
        self.CheckType(result, StateVector)
        
        # check solution tolerance
        if relTol < 0.0:
            raise ValueError('UserMemory.SolveAdjoint() >> ' + \
                             'Negative tolerance.')
                             
        # send the request to the user
        cost = self.userObj.solve_linearsys(atDesign.GetIndex(), 
                                            atState.GetIndex(),
                                            rhsVec.GetIndex(), tol,
                                            result.GetIndex())
                                            
        # update cost tracking and return convergence info
        self.precondCount += abs(cost)
        if cost < 0:
            return False
        else:
            return True
            
    def ObjectiveGradient(self, atDesign, atState, result):
        # check vector types
        self.CheckType(atDesign, DesignVector)
        self.CheckType(atState, StateVector)
        
        # send the request to the user
        if isinstance(result, DesignVector):
            # if result is a design vector, we want dJ/dDesign
            self.userObj.eval_grad_d(atDesign.GetIndex(), 
                                     atState.GetIndex(),
                                     result.GetIndex())
        elif isinstance(result, StateVector):
            # if result is a state vector, we want dJ/dState
            self.userObj.eval_grad_s(atDesign.GetIndex(), 
                                     atState.GetIndex(),
                                     result.GetIndex())
        else:
            # result vector is improper type
            raise TypeError('UserMemory.ObjectiveGradient() >> ' + \
                            'Wrong gradient type requested.')
    
    def GetRank():
        return self.userObj.get_rank()
        
    def CurrentDesign(self, design, state, adjoint, dual, optIter):
        # check vector types
        self.CheckType(design, DesignVector)
        self.CheckType(state, StateVector)
        self.CheckType(adjoint, StateVector)
        self.CheckType(dual, DualVector)
        
        # send the request to the user
        self.userObj.user_info(design.GetIndex(), state.GetIndex(),
                               adjoint.GetIndex(), dual.GetIndex(),
                               optIter)
                               
        # print cost information
        if self.GetRank() == 0:
            print "total preconditioner calls (UserMemory says) ="%self.precondCount
        
        
        
        
        