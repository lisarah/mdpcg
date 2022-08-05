# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:59:58 2019

@author: Sarah Li
"""
import util.utilities as ut
import util.cvx_util as cutil
import cvxpy as cvx

import models.mdpcg as mdpcg
class cvx_solver(mdpcg.quad_game):
    # #--------------- CVX Parameters  ---------------------
    y_ijt = {}
    objective_value = None
    exactPenalty = None
    #----------- Constraints --------------------
    positivity = None
    initialCondition = None
    massConservation = None
    constrainedState = None
    stateLB = None
    stateUB = None
    stateConstraints = None
    #-------exact penalty parameters--------------
    epsilon = 0.1 # for exact penalty
    optimalDual = None
#------------ initialize solver ----------------------------
    def __init__(self, time_steps = 20, manhattan = False):
        super().__init__(time_steps, manhattan)
        for i in range(self.States):
            for j in range(self.Actions):
                for t in range(self.Time):
                    self.y_ijt[(i,j,t)] = cvx.Variable()
 
#------------ set a constrained state-----------------------------
    def setConstrainedState(self, constrainedState, bound = None, 
                            isLB = True, verbose = False):
        self.constrainedState = constrainedState
        if isLB:
            if verbose:
                print ("lower bound set to" , bound)
            self.stateLB = bound
        else:
            if verbose:
                print ("upper bound set to", bound)
            self.stateUB = bound
        return True

# ----------------------LP create penalty --------------------------------------
    def penalty(self):
        y_ijt = self.y_ijt
        #NOTE: only one of the LB or UB is on at a time
        # This is toll corresponding to lower bound constraints imposed between time t = 3 and t = T. 
        if self.stateLB != None:
           # print self.Time
            objF = sum([(self.optimalDual[t-3] + self.epsilon) *
                               (self.stateLB - sum([y_ijt[(self.constrainedState,j,t)] 
                                       for j in range(self.Actions)]) )
                         for t in range(3, self.Time)])
        elif self.stateUB != None:
        # This is toll corresponding to upperbound constraints imposed for all time 
            objF = sum([(self.optimalDual[t-3] + self.epsilon)*
                               cvx.pos(sum([y_ijt[(self.constrainedState,j,t)] for j in range(self.Actions)])
                                - self.stateUB)
                         for t in range(self.Time)])
        self.exactPenalty = objF
# ----------------------LP setPositivity Constraints --------------------------
    def setPositivity(self):
        self.positivity =[]
        actions = self.Actions
        states = self.States
        time = self.Time
        for i in range(states):                
            for t in range(time):                     
                for j in range(actions):
                    # positivity constraints
                    self.positivity.append(self.y_ijt[(i,j,t)] >= 0.)
                    
# ----------------------LP set initial condition Constraints ------------------
    def setInitialCondition(self,p0):
        self.initialCondition = []
        actions = self.Actions
        states = self.States
        for i in range(states):
            # Enforce initial condition
            initState = sum([self.y_ijt[(i,j,0)] for j in range(actions)])
            if p0 is None:
                print ("no initial condition")
                if i == 0:
                    self.initialCondition.append(initState == 1.)
                else:
                    self.initialCondition.append(initState == 0.)
            else: 
                self.initialCondition.append(initState == p0[i])   
# ----------------------LP set mass conservation Constraints ------------------
    def setMassConservation(self):
        self.massConservation = []
        actions = self.Actions
        states = self.States
        time = self.Time
        for i in range(states):
            for t in range(time-1):  
                # mass conservation constraints between timesteps
                prevProb = sum([sum([
                    self.y_ijt[(iLast,j,t)] * self.P[t,i,iLast,j] 
                                for iLast in range(states) ]) 
                           for j in range(actions)]) 
                newProb = sum([self.y_ijt[(i,j,t+1)] 
                          for j in range(actions)])
                self.massConservation.append(newProb == prevProb)
# ----------------------LP set state Constraints ------------------------------
    def setStateConstraints(self, constraintList):
        self.stateConstraints = []
        for ind in range(len(constraintList)):
            con = constraintList[ind]
            constrainedState = self.y_ijt[(con.index)]
            if con.upperBound:
                self.stateConstraints.append(constrainedState <= con.value)
            else: 
                self.stateConstraints.append(constrainedState >= con.value)
#------------- CVX solver ----------------                    
    def solve(self, p0,  withPenalty = False, verbose = False,  
              returnDual= False, isSocial = False):        
        states = self.States
        actions = self.Actions
        time = self.Time
        if not self.objective_value:
            if verbose:
                print ("objective is set")
            self.objective_value = self.get_objective(isSocial)
        lp = None    
        # construct LP objective  
        if withPenalty:
            if self.exactPenalty is None:
                self.penalty()
            lp = cvx.Minimize(self.objective_value +  self.exactPenalty) # set lp problem            
        else:
            if verbose:
                print ("not with penalty")
            lp = cvx.Minimize(self.objective_value)


        if self.positivity is None:
            if verbose:
                print ("set positivity")
            self.setPositivity()
        if self.massConservation is None:
            if verbose:
                print ("set mass conservation")
            self.setMassConservation()
        if self.initialCondition is None:
            if verbose:
                print ("set initial condition")
            self.setInitialCondition(p0)
        
        mdpPolicy = cvx.Problem(lp, self.positivity+
                                    self.massConservation+
                                    self.initialCondition)
        
        mdpRes = mdpPolicy.solve(solver=cvx.ECOS, verbose=verbose)
        print (mdpRes)
        optRes = cutil.cvxDict2Arr(self.y_ijt,[states,actions,time])
        
        if returnDual:
            self.optimalDual = cutil.cvxList2Arr(self.massConservation,[states,time-1],returnDual)
            return optRes,self.optimalDual
        else:
            return optRes, mdpRes
        
#------------------- solve MDP with explicit constraints ----------------------
    def solveWithConstraint(self,
                            p0, 
                            verbose = False, 
                            constraintList = []):  
        states = self.States
        actions = self.Actions
        time = self.Time
        constrainedState = self.constrainedState
        if self.objective_value is None:
            if verbose:
                print ("setting constraints again")
            self.get_objective()
        lp = cvx.Minimize(self.objective_value)
        # construct constraints
        if self.positivity is None:
            self.setPositivity()
        if self.massConservation is None:
            self.setMassConservation()
        if self.initialCondition is None:
            self.setInitialCondition(p0)
        if len(constraintList) == 0:    
            self.stateConstraints = []
            # EXTRA DENSITY CONSTRAINT on constrained state  
            if self.stateLB != None:
                for t in range(3, time):
                    self.stateConstraints.append(sum([self.y_ijt[(constrainedState,j,t)] 
                                              for j in range(actions)])  
                                              >= self.stateLB) 
            elif self.stateUB != None:   
                for t in range(3, time):
                    self.stateConstraints.append(sum([self.y_ijt[(constrainedState,j,t)] 
                                              for j in range(actions)])  
                                              <= self.stateUB) 
   
        else: 
            self.setStateConstraints(constraintList)
            
        mdpPolicy = cvx.Problem(lp, self.positivity+
                                    self.massConservation+
                                    self.initialCondition+
                                    self.stateConstraints)
    
        
        mdpRes = mdpPolicy.solve(solver=cvx.ECOS, verbose=verbose)
        
        print (mdpRes)
        optRes = cutil.cvxDict2Arr(self.y_ijt,[states,actions,time])
        optDual = cutil.cvxList2Arr(self.stateConstraints,[len(self.stateConstraints)],True)
        self.optimalDual = ut.truncate(optDual)   
        return optRes    