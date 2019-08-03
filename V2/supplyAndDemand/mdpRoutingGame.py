# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:37:47 2018

@author: craba
"""


import mdp as mdp
import dynamicProgramming as dp
import figureGeneration as fG

import cvxpy as cvx
import numpy as np
import networkx as nx
import scipy.linalg as sla
import matplotlib.pyplot as plt


from collections import namedtuple
# specify which graph we are using for the mdp problem
gParam = namedtuple("gParam", "type rowSize colSize")

class mdpRoutingGame:
#--------------constructor-------------------------------
    def __init__(self, graph, Time):
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
        if graph.type is "seattleQuad":
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
                self.R[:,:,t] = 1.0*d + 1;
                self.C[:,:,t] = 1.0*c - c.min()*1.1;


                
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
        elif var is "States":
            return self.States;
        elif var is "Actions":
            return self.Actions;
        else:
            return "No proper variable was specified";
####################### SETTERS ###############################################
#------------ set MDP to quadratic Objective-----------------------------
    def setQuad(self):  
        self.isQuad = True;
#------------ set a constrained state-----------------------------
    def setConstrainedState(self, constrainedState, bound = None, isLB = True):
        self.constrainedState = constrainedState;
        if isLB:
            print "lower bound set to" , bound
            self.stateLB = bound;
        else:
            print "upper bound set to", bound;
            self.stateUB = bound;
        return True;
#-------------LP Obejective and Constraints --------------------
    def setObjective(self):
        y_ijt = {};
        for i in range(self.States):
            for j in range(self.Actions):
                for t in range(self.Time):
                    y_ijt[(i,j,t)] = cvx.Variable();

        objF = sum([sum([sum([0.5*cvx.pos(self.R[i,j,t])*cvx.square(y_ijt[(i,j,t)])
                     for i in range(self.States) ]) 
                for j in range(self.Actions)]) 
           for t in range(self.Time)]) \
               + sum([sum([sum([(self.C[i,j,t])*y_ijt[(i,j,t)]
                     for i in range(self.States) ]) 
                for j in range(self.Actions)]) 
           for t in range(self.Time)]);

        self.yijt = y_ijt;
        self.lpObj = objF;
# ----------------------LP create penalty --------------------------------------
    def penalty(self):
        y_ijt = self.yijt;
#        perActionPenalty = self.stateLB / self.Actions;
        
        #NOTE: only one of the LB or UB is on at a time
        # This is toll corresponding to lower bound constraints imposed between time t = 3 and t = T. 
        if self.stateLB != None:
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
                    
####################### MDP SOLVERS ###########################################       
#------------- unconstrained MDP solver (with exact penalty)  ----------------                    
    def solve(self,
              p0, 
              withPenalty = False,
              verbose = False, 
              returnDual= False):        
        states = self.States;
        actions = self.Actions;
        time = self.Time;
        if self.lpObj is None:
            print "objective is set"
            self.setObjective();
        lp = None;    
        # construct LP objective  
        if withPenalty:
            if self.exactPenalty is None:
                self.penalty();
            lp = cvx.Minimize(self.lpObj +  self.exactPenalty); # set lp problem            
        else:
            print "not with penalty"
            lp = cvx.Minimize(self.lpObj);
            
        y_ijt = self.yijt;

        if self.positivity is None:
            print "set positivity"
            self.setPositivity();
        if self.massConservation is None:
            print "set mass conservation"
            self.setMassConservation();
        if self.initialCondition is None:
            print "set initial condition"
            self.setInitialCondition(p0);
        
        mdpPolicy = cvx.Problem(lp, self.positivity+
                                    self.massConservation+
                                    self.initialCondition);
        
        mdpRes = mdpPolicy.solve(verbose=verbose)
        
        print mdpRes
        optRes = mdp.cvxDict2Arr(y_ijt,[states,actions,time]);
       
        
        if returnDual:
            self.optimalDual = mdp.cvxList2Arr(self.massConservation,[states,time-1],returnDual);
            return optRes,self.optimalDual;
        else:
            return optRes;
        
#------------------- solve MDP with explicit constraints ----------------------
    def solveWithConstraint(self,
                            p0, 
                            verbose = False):  
        states = self.States;
        actions = self.Actions;
        time = self.Time;
        constrainedState = self.constrainedState;

        
        if self.lpObj is None:
            print "setting constraints again"
            self.setObjective();
        lp = cvx.Maximize(self.lpObj);
        y_ijt = self.yijt;
        # construct constraints
        densityConstraints = [];
        if self.positivity is None:
            self.setPositivity();
        if self.massConservation is None:
            self.setMassConservation();
        if self.initialCondition is None:
            self.setInitialCondition(p0);
               
        # EXTRA DENSITY CONSTRAINT on constrained state  
        if self.isQuad:
            if self.stateLB != None:
                for t in range(3, time):
                    densityConstraints.append(sum([y_ijt[(constrainedState,j,t)] 
                                              for j in range(actions)])  
                                              >= self.stateLB); 
            elif self.stateUB != None:   
                for t in range(3, time):
                    densityConstraints.append(sum([y_ijt[(constrainedState,j,t)] 
                                              for j in range(actions)])  
                                              <= self.stateUB); 
        else:
            for t in range(time):
                densityConstraints.append(sum([y_ijt[(constrainedState,j,t)] 
                                          for j in range(actions)])  
                                          >= self.stateLB);            
        mdpPolicy = cvx.Problem(lp, self.positivity+
                                    self.massConservation+
                                    self.initialCondition+
                                    densityConstraints);
    
        
        mdpRes = mdpPolicy.solve(verbose=verbose)
        
        print mdpRes
        optRes = mdp.cvxDict2Arr(y_ijt,[states,actions,time]);
        optDual = mdp.cvxList2Arr(densityConstraints,[time],True);
        self.optimalDual = abs(mdp.truncate(optDual));   
        return optRes;
        
        
        
        
        
        
        
        
        
        
        
        
        
        