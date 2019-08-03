 # -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:37:47 2018

@author: craba
"""

import util.mdp as mdp
import util.figureGeneration as fG
import util.utilities as ut
import cvxpy as cvx
import numpy as np
import networkx as nx


from collections import namedtuple
# specify which graph we are using for the mdp problem
gParam = namedtuple("gParam", "type rowSize colSize")

class mdpRoutingGame:
#--------------constructor-------------------------------
    def __init__(self, graph, Time, strictlyConvex = True):
        #------------------------ MDP problem Parameters ----------------
        self.isQuad = False;
        self.R = None; # rewards matrix 
        self.C = None; # constant part of the reward
        self.P = None;
        self.Time = Time; # number of time steps
        self.States = None; # number of states
        self.Actions = None; # number of actions
        self.constrainedState = None;
        self.stateLB = None;
        self.stateUB = None;
        self.stateConstraints = None;
        #------------ Underlying Network parameters -------------------
        self.G = None;
        self.graphPos = None; # for drawing
        #--------------- LP CVX Parameters ---------------------
        self.yijt = None;
        self.lpObj = None;
        self.exactPenalty = None;
            #----------- LP Constraints--------------------
        self.positivity = None;
        self.initialCondition = None;
        self.massConservation = None;
        
        #-------exact penalty parameters--------------
        self.epsilon = 0.1; # for exact penalty
        self.optimalDual = None;
        
        # ---------- choose type of underlying graph---------------------------
        if graph.type is "grid":
            self.G = nx.grid_graph([graph.rowSize,graph.colSize]);
            self.G = nx.convert_node_labels_to_integers(self.G)
            self.graphPos=nx.spring_layout(self.G);
            self.States = graph.rowSize * graph.colSize;
            self.Actions = 5; # up, down left, right, stay
            self.P, c = mdp.generateGridMDP(self.States,
                                            self.Actions,
                                            self.G,
                                            test = True);
            self.R = np.zeros((self.States,self.Actions,Time));
            self.stateLB = 0.2;
            for t in range(Time):
                self.R[:,:,t] = 1.0*c;
        # seattle graph
        elif graph.type is "seattle":
            self.graphPos, self.G =  fG.NeighbourGen(False);
            self.States = self.G.number_of_nodes();
            self.Actions = len(nx.degree_histogram(self.G));
            self.P, c = mdp.generateMDP(self.States,
                                        self.Actions,
                                        self.G)
            self.R = np.zeros((self.States,self.Actions,Time));
            for t in range(Time):
                self.R[:,:,t] = 1.0*c;
        # seattle graph
        elif graph.type is "seattleQuad":
            self.graphPos, self.G, distances =  fG.NeighbourGen(False);
            self.States = self.G.number_of_nodes();
            self.Actions = len(nx.degree_histogram(self.G));
            self.P, c, d = mdp.generateQuadMDP(self.States,
                                        self.Actions,
                                        self.G,
                                        distances)
            self.R = np.zeros((self.States,self.Actions,Time));
            self.C = np.zeros((self.States,self.Actions,Time));
            for t in range(Time):
                if strictlyConvex:
                    self.R[:,:,t] = 1.0*d + 1;
                    self.C[:,:,t] = 1.0*c - c.min()*1.1;
                else:
                    self.R[:,:,t] = 1.0*d;
                    self.C[:,:,t] = 1.0*c;
                
                
######################## GETTER ###############################################
    def __call__(self,var): # return something
        # What to use this for
        if var is "optDual":
            return self.optimalDual;
        elif var is "reward":
            return self.R;
        elif var is "C":
            return self.C;
        elif var is "constrainedState":
            return self.constrainedState;
        elif var is "lowerBound":
            return self.stateLB;
        elif var is "probability":
            return self.P;
        elif var is "graphPos":
            return self.graphPos;
        elif var is "G":
            return self.G;
        elif var is "isQuad":
            return self.isQuad;
        else:
            return "No proper variable was specified";
####################### SETTERS ###############################################
#------------ set MDP to quadratic Objective-----------------------------
    def setQuad(self):  
        self.isQuad = True;
#------------ set a constrained state-----------------------------
    def setConstrainedState(self, constrainedState, bound = None, 
                            isLB = True, verbose = False):
        self.constrainedState = constrainedState;
        if isLB:
            if verbose:
                print "lower bound set to" , bound
            self.stateLB = bound;
        else:
            if verbose:
                print "upper bound set to", bound;
            self.stateUB = bound;
        return True;
#-------------LP Obejective and Constraints --------------------
    def setObjective(self,isSocial = False, verbose = False):
        y_ijt = {};
        for i in range(self.States):
            for j in range(self.Actions):
                for t in range(self.Time):
                    y_ijt[(i,j,t)] = cvx.Variable();
        if self.isQuad:
            if verbose:
                print "quadratic objective"
            if isSocial:
                objF = sum([sum([sum([-cvx.pos(self.R[i,j,t])*cvx.square(y_ijt[(i,j,t)])
                             for i in range(self.States) ]) 
                        for j in range(self.Actions)]) 
                   for t in range(self.Time)]) \
                       + sum([sum([sum([(self.C[i,j,t])*y_ijt[(i,j,t)]
                             for i in range(self.States) ]) 
                        for j in range(self.Actions)]) 
                   for t in range(self.Time)]);
            else:
                objF = sum([sum([sum([-0.5*cvx.pos(self.R[i,j,t])*cvx.square(y_ijt[(i,j,t)])
                             for i in range(self.States) ]) 
                        for j in range(self.Actions)]) 
                   for t in range(self.Time)]) \
                       + sum([sum([sum([(self.C[i,j,t])*y_ijt[(i,j,t)]
                             for i in range(self.States) ]) 
                        for j in range(self.Actions)]) 
                   for t in range(self.Time)]);
        else:
            objF = -sum([sum([sum([y_ijt[(i,j,t)]*self.R[i,j,t] 
                             for i in range(self.States) ]) 
                        for j in range(self.Actions)]) 
                   for t in range(self.Time)]);
        self.yijt = y_ijt;
        self.lpObj = objF;
# ----------------------LP create penalty --------------------------------------
    def penalty(self):
        y_ijt = self.yijt;
        #NOTE: only one of the LB or UB is on at a time
        # This is toll corresponding to lower bound constraints imposed between time t = 3 and t = T. 
        if self.stateLB != None:
#            print self.Time;
            objF = -sum([(self.optimalDual[t-3] + self.epsilon)*
                               cvx.pos(self.stateLB - sum([y_ijt[(self.constrainedState,j,t)] 
                                       for j in range(self.Actions)]) )
                         for t in range(3, self.Time)]);
        elif self.stateUB != None:
        # This is toll corresponding to upperbound constraints imposed for all time 
            objF = -sum([(self.optimalDual[t-3] + self.epsilon)*
                               cvx.pos(sum([y_ijt[(self.constrainedState,j,t)] for j in range(self.Actions)])
                                - self.stateUB)
                         for t in range(self.Time)])
        self.exactPenalty = objF;
# ----------------------LP setPositivity Constraints --------------------------
    def setPositivity(self):
        self.positivity =[];
        actions = self.Actions;
        states = self.States;
        time = self.Time;
        y_ijt = self.yijt;
        for i in range(states):                
            for t in range(time):                     
                for j in range(actions):
                    # positivity constraints
                    self.positivity.append(y_ijt[(i,j,t)] >= 0.)
                    
# ----------------------LP set initial condition Constraints ------------------
    def setInitialCondition(self,p0):
        self.initialCondition = [];
        actions = self.Actions;
        states = self.States;
        y_ijt = self.yijt;  
        for i in range(states):
            # Enforce initial condition
            initState = sum([y_ijt[(i,j,0)] for j in range(actions)]);
            if p0 is None:
                print "no initial condition";
                if i == 0:
                    self.initialCondition.append(initState == 1.)
                else:
                    self.initialCondition.append(initState == 0.)
            else: 
                self.initialCondition.append(initState == p0[i]);   
# ----------------------LP set mass conservation Constraints ------------------
    def setMassConservation(self):
        self.massConservation = [];
        actions = self.Actions;
        states = self.States;
        time = self.Time;
        y_ijt = self.yijt;
        for i in range(states):
            for t in range(time-1):  
                # mass conservation constraints between timesteps
                prevProb = sum([sum([y_ijt[(iLast,j,t)]*self.P[i,iLast,j] 
                                for iLast in range(states) ]) 
                           for j in range(actions)]) ;
                newProb = sum([y_ijt[(i,j,t+1)] 
                          for j in range(actions)]);
                self.massConservation.append(newProb == prevProb);
# ----------------------LP set state Constraints ------------------------------
    def setStateConstraints(self, constraintList):
        y_ijt = self.yijt;
        self.stateConstraints = [];
        for ind in range(len(constraintList)):
            con = constraintList[ind];
            constrainedState = y_ijt[(con.index)];
            if con.upperBound:
                self.stateConstraints.append(constrainedState <= con.value)
            else: 
                self.stateConstraints.append(constrainedState >= con.value)
        

                    
####################### MDP SOLVERS ###########################################       
#------------- unconstrained MDP solver (with exact penalty)  ----------------                    
    def solve(self,
              p0, 
              withPenalty = False,
              verbose = False, 
              returnDual= False,
              isSocial = False):        
        states = self.States;
        actions = self.Actions;
        time = self.Time;
        if self.lpObj is None:
            if verbose:
                print "objective is set"
            self.setObjective(isSocial);
        lp = None;    
        # construct LP objective  
        if withPenalty:
            if self.exactPenalty is None:
                self.penalty();
            lp = cvx.Maximize(self.lpObj +  self.exactPenalty); # set lp problem            
        else:
            if verbose:
                print "not with penalty"
            lp = cvx.Maximize(self.lpObj);
            
        y_ijt = self.yijt;

        if self.positivity is None:
            if verbose:
                print "set positivity"
            self.setPositivity();
        if self.massConservation is None:
            if verbose:
                print "set mass conservation"
            self.setMassConservation();
        if self.initialCondition is None:
            if verbose:
                print "set initial condition"
            self.setInitialCondition(p0);
        
        mdpPolicy = cvx.Problem(lp, self.positivity+
                                    self.massConservation+
                                    self.initialCondition);
        
        mdpRes = mdpPolicy.solve(solver=cvx.ECOS, verbose=verbose)
        
        print mdpRes
        optRes = mdp.cvxDict2Arr(y_ijt,[states,actions,time]);
       
        
        if returnDual:
            self.optimalDual = mdp.cvxList2Arr(self.massConservation,[states,time-1],returnDual);
            return optRes,self.optimalDual;
        else:
            return optRes, mdpRes;
        
#------------------- solve MDP with explicit constraints ----------------------
    def solveWithConstraint(self,
                            p0, 
                            verbose = False, 
                            constraintList = []):  
        states = self.States;
        actions = self.Actions;
        time = self.Time;
        constrainedState = self.constrainedState;

        
        if self.lpObj is None:
            if verbose:
                print "setting constraints again"
            self.setObjective();
        lp = cvx.Maximize(self.lpObj);
        y_ijt = self.yijt;
        # construct constraints
        if self.positivity is None:
            self.setPositivity();
        if self.massConservation is None:
            self.setMassConservation();
        if self.initialCondition is None:
            self.setInitialCondition(p0);
        if len(constraintList) == 0:    
            self.stateConstraints = [];
            # EXTRA DENSITY CONSTRAINT on constrained state  
            if self.isQuad:
                if self.stateLB != None:
                    for t in range(3, time):
                        self.stateConstraints.append(sum([y_ijt[(constrainedState,j,t)] 
                                                  for j in range(actions)])  
                                                  >= self.stateLB); 
                elif self.stateUB != None:   
                    for t in range(3, time):
                        self.stateConstraints.append(sum([y_ijt[(constrainedState,j,t)] 
                                                  for j in range(actions)])  
                                                  <= self.stateUB); 
            else:
                for t in range(time):
                    self.stateConstraints.append(sum([y_ijt[(constrainedState,j,t)] 
                                              for j in range(actions)]) 
                                          >= self.stateLB);     
        else: 
            self.setStateConstraints(constraintList);
            
        mdpPolicy = cvx.Problem(lp, self.positivity+
                                    self.massConservation+
                                    self.initialCondition+
                                    self.stateConstraints);
    
        
        mdpRes = mdpPolicy.solve(solver=cvx.ECOS, verbose=verbose)
        
        print mdpRes
        optRes = mdp.cvxDict2Arr(y_ijt,[states,actions,time]);
        optDual = mdp.cvxList2Arr(self.stateConstraints,[len(self.stateConstraints)],True);
        self.optimalDual = ut.truncate(optDual);   
        return optRes;
        
        
        
    def socialCost(self, trajectory):
        objF = sum([sum([sum([-self.R[i,j,t]*trajectory[i,j,t]*trajectory[i,j,t]
                         for i in range(self.States) ]) 
                    for j in range(self.Actions)]) 
               for t in range(self.Time)]) \
               + sum([sum([sum([(self.C[i,j,t])*trajectory[i,j,t]
                           for i in range(self.States) ]) 
                    for j in range(self.Actions)]) 
                  for t in range(self.Time)])
        return objF;
        
        
        
        
        
        
        
        
        