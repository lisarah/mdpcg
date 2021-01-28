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
        for s in range(len(self.states)):
            y_sa_list.append([])
            for a in range(len(self.actions)):
                y_sa_list[-1].append(cvx.Variable())
        self.y_sa = np.array(y_sa_list)
        w_list = [cvx.Variable() for  c in self.collision_set]
        self.w  = np.array(w_list)
        self.constraints = []
        self.objective = None
    def set_constraints(self):     
        for s in self.states.values():
            # set initial distribution
            self.constraints.append(
                sum([self.y_sa[s,a] for a in self.actions.values()]) == 1)
            # set positivity
            for a in self.actions.values():
                self.constraints.append(self.y_sa[s,a] >= 0)
                
        for a in self.actions.values():
        # set final distribution
            self.constraints.append(
                sum([self.y_sa[s,a] for s in self.states.values()]) == 1)
        
        # w_ind = 0
        # for c in self.collision_set:
        # # set relaxation of bilinear constraints
        #     x = self.y_sa[c[0], c[1]]
        #     y = self.y_sa[c[2], c[3]]
        #     self.constraints.append(self.w[w_ind] >= x + y)
        #     self.constraints.append(self.w[w_ind] <= y)
        #     self.constraints.append(self.w[w_ind] <= x)
        #     w_ind += 1
   
    def solve(self,  returnDual= True, verbose = False):
        self.objective = self.set_objective(self.y_sa, self.w)
        self.set_constraints()
        cvx_objective = cvx.Minimize(self.objective)
        cvx_problem= cvx.Problem(cvx_objective, self.constraints)
        cvx_solution = cvx_problem.solve(verbose=verbose) # solver=cvx.ECOS, 
        
        # print (f'minimized total cost is {cvx_solution}')
        optRes = cutil.cvx_array_2_array(self.y_sa)
        print (f'cvx minimized distribution \n{np.round(optRes, 2)}')
        return optRes
        