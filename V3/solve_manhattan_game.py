# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:07:53 2021

@author: Sarah Li
"""
import numpy as np
import gameSolvers.cvx as cvx
import algorithm.FW as fw
import models.mdpcg as mdpcg
import models.taxi_dynamics.manhattan_transition as m_dynamics
import models.taxi_dynamics.manhattan_neighbors as m_neighbors
import models.taxi_dynamics.visualization as visual
import util.plot_lib as pt

T = 15
epsilon = 1e-1
borough = 'Manhattan'
manhattan_game = mdpcg.quad_game(T, manhattan=True)
initial_distribution = m_dynamics.uniform_initial_distribution(10000)
#----- CVX version ------ currently isn't running
# manhattan_game.solve(driver_distribution, verbose=True, returnDual=False)



x0 = np.zeros((manhattan_game.States, manhattan_game.Actions,
               manhattan_game.Time));   
y_opt, y_history = fw.FW(x0, initial_distribution, manhattan_game.P, 
                         manhattan_game.evaluate_cost, False, epsilon, 
                         maxIterations = 1000);
obj_history = [];
for i in range(len(y_history)):
	obj_history.append(manhattan_game.evaluate_objective(y_history[i]))

print(f'Last objective value is {obj_history[-1]}')
# pt.latex_format()
optimal_value = 0
pt.objective(obj_history, optimal_value, 'Frank Wolfe')

#-----------visualize optimal distribution ------------------#
density_dict = {}
state_ind  = m_neighbors.zone_to_state(m_neighbors.zone_neighbors)
for zone_ind in m_neighbors.zone_neighbors.keys():
    density_dict[zone_ind] = np.sum(y_opt[state_ind[zone_ind], :, T-1])
visual.animate_borough(borough, density_dict)