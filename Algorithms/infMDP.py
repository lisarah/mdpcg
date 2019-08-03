# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:07:05 2018

@author: craba
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:25:24 2018

@author: craba
"""

import util.mdp as mdp
import util.figureGeneration as fG

import cvxpy as cvx
import numpy as np
import networkx as nx



from collections import namedtuple
# specify which graph we are using for the mdp problem
gParam = namedtuple("gParam", "type rowSize colSize")

class infMDP:
#--------------constructor-------------------------------
    def __init__(self, graph, beta):
        #------------------------ MDP problem Parameters ----------------
        self.R = None; # rewards matrix 
        self.C = None; # constant part of the reward
        self.P = None;
        self.States = None; # number of states
        self.Actions = None; # number of actions
        self.beta = beta;
        #------------ Underlying Network parameters -------------------
        self.G = None;
        self.graphPos = None; # for drawing
        #--------------- LP CVX Parameters ---------------------
        self.yij = None;
        self.obj = None;
            #----------- LP Constraints--------------------
        self.positivity = None;
        self.massConservation = None;        
        # ---------- choose type of underlying graph---------------------------
        # seattle graph
        if graph.type is "seattleQuad":
            self.graphPos, self.G, distances =  fG.NeighbourGen(False);
            self.States = self.G.number_of_nodes();
            self.Actions = len(nx.degree_histogram(self.G));
            self.P, c, d = mdp.generateQuadMDP(self.States,
                                        self.Actions,
                                        self.G,
                                        distances,
                                        p = 0.7)
            self.R = np.zeros((self.States,self.Actions));
            self.C = np.zeros((self.States,self.Actions));
            self.R = 1.0*d;
            self.C = 1.0*c;
                
####################### SETTERS ###############################################
#-------------LP Obejective and Constraints --------------------
    def setObjective(self):
        y_ij = {};
        for i in range(self.States):
            for j in range(self.Actions):
                    y_ij[(i,j)] = cvx.Variable();

        objF = sum([sum([-0.5*cvx.pos(self.R[i,j])*cvx.square(y_ij[(i,j)])
                     for i in range(self.States) ]) 
                for j in range(self.Actions)])/(1.-self.beta) \
               + sum([sum([(self.C[i,j])*y_ij[(i,j)]
                     for i in range(self.States) ])
                for j in range(self.Actions)]);
        self.yij = y_ij;
        self.lpObj = 1/self.beta*objF;

# ----------------------LP setPositivity Constraints --------------------------
    def setPositivity(self):
        self.positivity =[];
        actions = self.Actions;
        states = self.States;
        y_ij = self.yij;
        for i in range(states):                                   
            for j in range(actions):
                # positivity constraints
                self.positivity.append(y_ij[(i,j)] >= 0.)
                    
# ----------------------LP set mass conservation Constraints ------------------
    def setMassConservation(self, p0):
        self.massConservation = [];
        actions = self.Actions;
        states = self.States;
        y_ij = self.yij;
        for i in range(states):
            # mass conservation constraints between timesteps
            prevProb = self.beta*sum([sum([y_ij[(iLast,j)]*self.P[i,iLast,j] 
                            for iLast in range(states) ]) 
                       for j in range(actions)]) + (1-self.beta)*p0[i] ;
            newProb = sum([y_ij[(i,j)] 
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
        self.setObjective();
        

        lp = cvx.Maximize(self.lpObj);
        y_ij = self.yij;
        print "set positivity"
        self.setPositivity();
        print "set mass conservation"
        self.setMassConservation(p0);
        
        mdpPolicy = cvx.Problem(lp, self.positivity+
                                    self.massConservation);
        
        mdpRes = mdpPolicy.solve(verbose=verbose)        
        print mdpRes
        optRes = mdp.cvxDict2Arr(y_ij,[states,actions]);
       
        return mdpRes, optRes;
            
        
        
        
        
        
        
        
        
        
        