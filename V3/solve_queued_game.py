# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:42:08 2022

@author: Sarah Li
"""
import algorithm.FW as fw
import models.queued_mdp_game as queued_game
import models.taxi_dynamics.visualization as visual


mass = 10000
# for debugging
# np.random.seed(49952574)
# print(f' current seed is {np.random.get_state()[1][0]}')
# np.random.seed(3239535799)
manhattan_game = queued_game.queue_game(mass, 0.01, uniform_density=True, 
                                        flat=False)
constrained_value = 350 
T = len(manhattan_game.forward_P)
initial_density = manhattan_game.get_density()
y_res, obj_hist = fw.FW_dict(manhattan_game, 
                             max_error=1000, max_iterations=1e3)
print(f'FW solved objective = {obj_hist[-1]}')

# get densities to visualize
z_density = manhattan_game.get_zone_densities(y_res[-1], False)
constraint_violation, violation_density = manhattan_game.get_violations(
    z_density, constrained_value)
avg_density = manhattan_game.get_average_density(z_density)

visual.summary_plot(z_density, constraint_violation, violation_density, 
                    avg_density, constrained_value)
