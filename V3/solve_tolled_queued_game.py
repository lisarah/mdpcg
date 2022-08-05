# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:31:31 2022

@author: Sarah Li
"""
import numpy as np
import algorithm.FW as fw
import algorithm.inexact_projected_gradient_ascent as pga
import models.queued_mdp_game as queued_game
import models.taxi_dynamics.manhattan_transition as m_dynamics
import models.taxi_dynamics.manhattan_neighbors as m_neighbors
import models.taxi_dynamics.visualization as visual
import util.plot_lib as pt
import util.utilities as ut
import matplotlib.pyplot as plt
import matplotlib as mpl



epsilon = 1000 # error in approximate solution of drivers learning
# epsilon_list = [5e3]  # [1e5, 2e4, 2e3, 1e3]
borough = 'Manhattan' # borough of interest
mass = 10000 # game population size
constrained_value = 300 # maximum driver density per state
max_error = 1000
max_iterations = 100 # number of iterations of dual ascent
toll_queues = False
# game definition with initial distribution
# print(f'cur seed {np.random.get_state[1][0]}')
np.random.seed(3239535799)
manhattan_game = queued_game.queue_game(mass, 0.01, uniform_density=True) 

initial_density = manhattan_game.whole_length_density()
y_res, obj_hist = fw.FW_dict(manhattan_game, max_error, max_iterations)
# determine the step size, which depends on strong convexity 
# and the constraint norm
alpha = manhattan_game.get_strong_convexity()
print(f'convexity factor is {alpha}')

# set constraints
# Z = len(manhattan_game.z_list)
T = len(manhattan_game.forward_P)
Z = 3
total_q = 8
constrained_zones = [161, 68, 237] #[161,43,68,79,231,236,237,114]# manhattan_game.z_list
constrained_states = [(c, 0) for c in constrained_zones]
total_violation = manhattan_game.get_constrained_gradient(
    y_res[-1], return_norm = True)
print(f'before tolling: violation_density = {total_violation}')
# if toll_queues:
#     q = 1
#     while q < total_q:
#         constrained_states = constrained_states + [(c, q) 
#                                                    for c in constrained_zones]

# for c in constrained_zones:
#     constrained_states = constrained_states + [(c, q) for q in range(total_q)]

# ZA = sum([len(manhattan_game.action_dict[(z, 0)]) for z in constrained_zones])
# # for z in constrained_zones:
# #     ZA += sum([len(manhattan_game.action_dict[(z, q)]) 
# #                for q in range(total_q)]) 
# T = len(manhattan_game.forward_P)

# A_arr = np.zeros((T*Z, ZA*T))  
# # sa_ind = 0
# za_ind = 0
# actions = manhattan_game.action_dict.values()
# for t in range(T):
#     z_ind = 0
#     for z in constrained_zones:
#         for a_ind in range(len(manhattan_game.action_dict[(z, 0)])):
#             A_arr[t*Z+z_ind, a_ind+za_ind] = 1
#         za_ind += len(manhattan_game.action_dict[(z, 0)])
#         z_ind += 1 
# two_norm_A = np.linalg.norm(A_arr,2)
# step_size = alpha/2/(two_norm_A**2)
# print(f'norm of A is {two_norm_A}, step size is {step_size}')
step_size = 0.0001
# set initial toll value
# determine each state action's constraint violation

# violation_density = {}
# total_violation = 0
# for z in constrained_zones:
#     state_density = [sum([y_res[-1][t][((z,0),a)] for a in manhattan_game.action_dict[(z,0)]])
#                      for t in range(T)]
#     violation_density[z] = [state_density[t] - constrained_value for t in range(T)]
#     for t in range(len(violation_density[z])):
#         if violation_density[z][t] < 0:
#             violation_density[z][t] = 0
#     total_violation += np.linalg.norm(np.array(violation_density[z]), 2)

        
tau = {}
for s in constrained_states:
    tau.update({(s, t): 0 for t in range(T)})
# step_size = 0.005

# find average constraint violation
average_tau = {z: 0 for z in tau.keys()}
avg_distribution = [{sa: 0 for sa in y_res[-1][t].keys()} 
                    for t in range(T)]
avg_violation = []
violation_state = constrained_zones[0]
distribution_history = []
social_cost = []
# define dual ascent approximate gradient update.
def approx_gradient(game, cur_tau, epsilon, k): # this eps comes from inexact_pga
    game.update_tolls(cur_tau)
    # solve first game
    approx_y, obj_hist = fw.FW_dict(game, max_error, max_iterations, 
                                    initial_density, verbose=False)
    
    # print(f'cost in violation state: {game.costs[10][((violation_state,0),7)]}')
    distribution_history.append(approx_y[-1])
    social_cost.append(game.get_social_cost(approx_y[-1]))

    # average tau
    for z in cur_tau.keys():
        average_tau[z] = (average_tau[z]*k+cur_tau[z])/(k+1) 
    # average distribution
    distribution_diff = 0
    for t in range(T):
        for sa in approx_y[-1][t].keys():
            avg_distribution[t][sa] = (avg_distribution[t][sa]*k+approx_y[-1][t][sa])/(k+1)
            distribution_diff += abs(avg_distribution[t][sa] - approx_y[-1][t][sa])
    print(f' {k}: distribution difference {distribution_diff}')
    gradient, violation = manhattan_game.get_constrained_gradient(
        approx_y[-1], return_violation=True)
    avg_violation.append(violation)
    print(f'{k}: violation {avg_violation[-1]}')
    population_sum = sum([sum([avg_distribution[t][sa] 
                               for sa in avg_distribution[t].keys()])
                          for t in range(T)])
    assert round(population_sum - T*mass, 5) == 0,  \
        f'population sum at {k}: {population_sum}'
    
    return gradient
    
# epsilons = [epsilon for i in range(max_iteration)]
tau_hist, gradient_hist = pga.inexact_pga(manhattan_game, tau, approx_gradient, step_size, 
                                          100, epsilons = [1000]*100000, 
                                          verbose = False)

# find average tau value
tau_values = []
average_tau = {z: 0 for z in tau_hist[-1].keys()}
tau_ind = 0
for tau_ind in range(len(tau_hist)):
    for z in tau.keys():
        average_tau[z] = (average_tau[z]*tau_ind+tau_hist[tau_ind][z])/(tau_ind+1) 
    tau_values.append(sum([abs(tau) for tau in average_tau.values()]))
    # tau_arr = np.array(list(average_tau.values())) 
    # tau_values.append(np.linalg.norm(tau_arr, 1))
# find average constraint violation

    
plt.figure()
plt.plot(tau_values)
plt.show()

#%% average tolling values for last approximation %%#
plt.figure()
print('fig 3 = toll norm as function of designer iteration')
plt.plot(tau_values, linewidth=3, label='total toll value')
plt.plot(avg_violation, label = 'total constraint violation', linewidth=3)
plt.legend()
plt.xlabel('Iterations')
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('toll_norm_convergence.png')


# social cost plot
pt.objective(social_cost, social_cost[0], 'Social Cost for driver')  
plt.tight_layout()
plt.savefig('grad_res/social_driver_profit.png')