# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:47:22 2021

Solving for the epsilon Wardrop equilibrium of the traffic MDPCG with 
frank wolfe.

@author: Sarah Li
"""
import numpy as np
import algorithm.FW as fw
import models.mdpcg as mdpcg
import util.plot_lib as pt
time_steps = 20
epsilon = 1e-1
game = mdpcg.quad_game(time_steps)
p0 = np.zeros((game.G.number_of_nodes()));
#p0[0] = 1.0;
# make all drivers start from residential areas 6 of them
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

optimal_value = 218094.34132103052 # as derived by cvxpy

x0 = np.zeros((game.States, game.Actions, game.Time));   
y_opt, y_history = fw.FW(x0, p0, game.P, game.evaluate_cost, False, epsilon);
obj_history = [];
for i in range(len(y_history)):
	obj_history.append(game.evaluate_objective(y_history[i]))

# pt.latex_format()
pt.objective(obj_history, optimal_value, 'Frank Wolfe')