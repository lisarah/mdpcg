# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 11:40:44 2021

CVX solver for the satellite game.
@author: Sarah Li
"""
import models.satellite_game as satellite_game
import util.cvx_util as cutil
import cvxpy as cvx
import numpy as np

class cvx_satellite(satellite_game.satellite_game):
    def __init__(self, r_init_set, r_final_set):
        super().__init__(r_init_set, r_final_set)
        y_sa_list = []
        for s in self.states.keys():
            y_sa_list.append([])
            for a in self.actions.keys():
                y_sa_list[-1].append(cvx.Variable())
        self.y_sa = np.array(y_sa_list)
        self.constraints = []
        self.objective = None
    def set_constraints(self):     
        for s in self.states.keys():
            # set initial distribution
            self.constraints.append(
                sum([self.y_sa[s,a] for a in self.actions.keys()]) == 1)
            # set positivity
            for a in self.actions.keys():
                self.constraints.append(self.y_sa[s,a] >= 0)
                
        for a in self.actions.keys():
        # set final distribution
            self.constraints.append(
                sum([self.y_sa[s,a] for s in self.states.keys()]) == 1)
   
    def solve(self,  returnDual= True, verbose = False):
        self.objective = self.set_objective(self.y_sa)
        self.set_constraints()
        cvx_objective = cvx.Minimize(self.objective)
        cvx_problem= cvx.Problem(cvx_objective, self.constraints)
        cvx_solution = cvx_problem.solve(verbose=verbose) # solver=cvx.ECOS, 
        
        print (f'minimized total cost is {cvx_solution}')
        optRes = cutil.cvx_array_2_array(self.y_sa)
        print (f'optimized assignment is \n{optRes}')
        return optRes
        