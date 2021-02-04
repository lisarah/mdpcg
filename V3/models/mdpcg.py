# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:58:25 2019

@author: Sarah Li
"""
import models.taxi_dynamics.manhattan_transition as m_transition
import models.taxi_dynamics.manhattan_cost as m_cost
import util.mdp as mdp
import util.figureGeneration as fg
import networkx as nx
import numpy as np
import cvxpy as cvx

class quad_game:
#--------------constructor-------------------------------
    def __init__(self, Time, manhattan = False, strictlyConvex = True):
        self.Time = Time
        self.verbose = True # debug verbosity.
        if manhattan:
            self.manhattan_gen(Time)
        else:  
            self.seattle_gen(Time, strictlyConvex)  
    def manhattan_gen(self, T):
        P_time_vary = m_transition.transition_kernel(T, 0.1)
        self.P = P_time_vary[0,:,:,:]
        m_transition.test_transition_kernel(P_time_vary)
        # random rider demand generation
        S, _, A = self.P.shape
        self.States = S
        self.Actions = A
        P_pick_up, demand_rate = m_transition.random_demand_generation(T, S)
        # the last action in P is for the action of trying to pick up drivers.
        self.P[:,:, A - 1] += P_pick_up[0,:,:]
        
        
        # cost generation
        self.R, self.C = m_cost.congestion_cost(demand_rate, T, S, A, 
                                                epsilon = 1e-3)
        
        
    def seattle_gen(self, Time, strictlyConvex):
        """TODO: move this somewhere else. Seattle MDP and cost generation.
        """
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
        self.P = P
        for t in range(Time):
            if strictlyConvex:
                self.R[:,:,t] = 1.0*d + 1;
                self.C[:,:,t] = 1.0*c - c.min()*1.1;
            else:
                self.R[:,:,t] = 1.0*d;
                self.C[:,:,t] = 1.0*c;
       
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
        toll_term = 0
        if constraint_value != 0:
            for toll in tolls:
                toll_term += 16 * 6 * constraint_value * toll

        return linear_term + 0.5 * quad_term + toll_term

    def evaluate_cost(self, y):
        """Evaluate cost at current population distribution."""
        return np.multiply(self.R, y) + self.C  - self.tolls
    def reset_toll(self):
        self.tolls = np.zeros((self.States,self.Actions,self.Time))