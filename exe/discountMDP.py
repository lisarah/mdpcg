# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:25:24 2018

@author: craba
"""

import util.mdp as mdp
import Algorithms.dynamicProgramming as dp
import util.figureGeneration as fG

import cvxpy as cvx
import numpy as np
import networkx as nx
import scipy.linalg as sla
import matplotlib.pyplot as plt


from collections import namedtuple
# specify which graph we are using for the mdp problem
gParam = namedtuple("gParam", "type rowSize colSize")

class discountMDP:
#--------------constructor-------------------------------
    def __init__(self, graph, Time, beta):
        #------------------------ MDP problem Parameters ----------------
        self.isQuad = False;
        self.R = None; # rewards matrix 
        self.C = None; # constant part of the reward
        self.P = None;
        self.States = None; # number of states
        self.Actions = None; # number of actions
        self.beta = 1.; # discount factor
        #------------ Underlying Network parameters -------------------
        self.G = None;
        self.graphPos = None; # for drawing
        #--------------- LP CVX Parameters ---------------------
        self.yij = None;
        self.lpObj = None;
            #----------- LP Constraints--------------------
        self.positivity = None;
        self.initialCondition = None;
        self.massConservation = None;
        
        #-------exact penalty parameters--------------
        self.epsilon = 0.1; # for exact penalty
        self.optimalDual = None;
        
        # ---------- choose type of underlying graph---------------------------
        # seattle graph
        if graph.type is "seattleQuad":
            self.graphPos, self.G, distances =  fG.NeighbourGen(False);
            self.States = self.G.number_of_nodes();
            self.Actions = len(nx.degree_histogram(self.G));
            self.Time = Time;
            self.P, c, d = mdp.generateQuadMDP(self.States,
                                        self.Actions,
                                        self.G,
                                        distances)
            self.R = np.zeros((self.States,self.Actions,Time));
            self.C = np.zeros((self.States,self.Actions,Time));
            for t in range(Time):
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
        elif var is "constrainedUpperBound":
            return self.constrainedUpperBound;
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
    def setConstrainedState(self, constrainedState, constrainedUpperBound = None):
        self.constrainedState = constrainedState;
        if constrainedUpperBound is not None:
            print "upperbound set to" , constrainedUpperBound
            self.constrainedUpperBound = constrainedUpperBound;
        return True;
#-------------LP Obejective and Constraints --------------------
    def setObjective(self):
        y_ijt = {};
        for i in range(self.States):
            for j in range(self.Actions):
                for t in range(self.Time):
                    y_ijt[(i,j,t)] = cvx.Variable();
        if self.isQuad:
            print "quadratic objective"
            objF = sum([sum([sum([-0.5*(self.beta**t)*cvx.pos(self.R[i,j,t])*cvx.square(y_ijt[(i,j,t)])
                         for i in range(self.States) ]) 
                    for j in range(self.Actions)]) 
               for t in range(self.Time)]) \
                   + sum([sum([sum([((self.beta**t)*self.C[i,j,t])*y_ijt[(i,j,t)]
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
              beta,
              withPenalty = False,
              verbose = False, 
              returnDual= False):        
        states = self.States;
        actions = self.Actions;
        time = self.Time;
        self.beta = beta;
        self.setObjective();
        

        lp = cvx.Maximize(self.lpObj);
        y_ijt = self.yijt;
        print "set positivity"
        self.setPositivity();
        print "set mass conservation"
        self.setMassConservation();
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
            return mdpRes, optRes;
        
        
        
        
        
        
        
        
        
        