# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:58:25 2019

@author: craba
"""
import util.mdp as mdp
import util.figureGeneration as fg
import networkx as nx
import numpy as np
import cvxpy as cvx
class quad_game:
#--------------constructor-------------------------------
    def __init__(self, Time, strictlyConvex = True):
        graphPos, G, distances =  fg.NeighbourGen(False);
        self.States = G.number_of_nodes();
        self.Actions = len(nx.degree_histogram(G));
        P, c, d = mdp.generateQuadMDP(self.States,
                                      self.Actions,
                                      G,
                                      distances)
        #------------------------ Game graph -----------------------
        self.G = G
        self.graphPos = graphPos
        #------------------------ MDP problem Parameters ----------------
        self.R = np.zeros((self.States, self.Actions,Time)); # quadratic part of cost 
        self.C = np.zeros((self.States,self.Actions,Time)); # constant part of the cost
        self.tolls = np.zeros((self.States,self.Actions,Time))
        for t in range(Time):
            if strictlyConvex:
                self.R[:,:,t] = 1.0*d + 1;
                self.C[:,:,t] = 1.0*c - c.min()*1.1;
            else:
                self.R[:,:,t] = 1.0*d;
                self.C[:,:,t] = 1.0*c;
        self.P = P;
        self.Time = Time; # number of time steps
        self.verbose = False; # debug parameter to output states       
            
    def get_objective(self, is_social = False):
        """ Define the objective using cvxpy variables."""
        y_ijt = self.y_ijt
        if self.verbose:
            print ("game set with quadratic objective")
        quad_term =  sum([sum([sum([
            cvx.pos(self.R[i, j, t]) * cvx.square(y_ijt[(i, j, t)])
                for i in range(self.States) ]) 
                for j in range(self.Actions)]) 
                for t in range(self.Time)]) 
        linear_term = sum([sum([sum([self.C[i, j, t]*y_ijt[(i, j, t)]
                for i in range(self.States) ]) 
                for j in range(self.Actions)]) 
                for t in range(self.Time)]);
        
        objective_value = 1 * linear_term
        if is_social:
            objective_value = objective_value + quad_term
        else:
            objective_value = objective_value + 0.5 * quad_term
        return objective_value;
    
    def evaluate_objective(self, y, constraint_value = 0, tolls = None):
        """Evaluate the objective for a given population distribution."""
        quad_term =  sum([sum([sum([self.R[i, j, t] * y[i, j, t]**2
            for i in range(self.States) ]) 
            for j in range(self.Actions)]) 
            for t in range(self.Time)]) 
        linear_term = sum([sum([sum([self.C[i, j, t] * y[i, j, t]
            for i in range(self.States) ]) 
            for j in range(self.Actions)]) 
            for t in range(self.Time)])
        if constraint_value != 0:
            for toll in tolls:
                obj += 16 * 6 * constraint_value * toll

        return linear_term + 0.5 * quad_term

    def evaluate_cost(self, y):
        """Evaluate cost at current population distribution."""
        return np.multiply(self.R, y) + self.C  - self.tolls
    def reset_toll(self):
        self.tolls = np.zeros((self.States,self.Actions,self.Time))