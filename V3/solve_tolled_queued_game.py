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
import models.test_model as test


epsilon = 100 # error in approximate solution of drivers learning
# epsilon_list = [5e3]  # [1e5, 2e4, 2e3, 1e3]
borough = 'Manhattan' # borough of interest
mass = 10000 # game population size
constrained_value = 350 # maximum driver density per state
max_error = 5000
max_iterations = 1000 # number of iterations of dual ascent
toll_queues = False
# game definition with initial distribution
# print(f'cur seed {np.random.get_state[1][0]}')
np.random.seed(3239535799)
manhattan_game = queued_game.queue_game(mass, 0.01, uniform_density=True,
                                        flat=True) 

initial_density = manhattan_game.get_density()
y_res, obj_hist = fw.FW_dict(manhattan_game, max_error, max_iterations)
# determine the step size, which depends on strong convexity 
# and the constraint norm
alpha = manhattan_game.get_strong_convexity()
print(f'convexity factor is {alpha}')

# set constraints
T = len(manhattan_game.forward_P)
constrained_zones = [161, 162, 236, 237] #[161,43,68,79,231,236,237,114]# manhattan_game.z_list
Z = len(constrained_zones)
# constrained_states = [(c, 0) for c in constrained_zones]
manhattan_game.set_constraints(constrained_zones, constrained_value)
grad, violation_norm = manhattan_game.get_constrained_gradient(
    y_res[-1], return_violation = True)
print(f'before tolling: violation_density = {violation_norm}')

ZA = sum([len(manhattan_game.action_dict[(z, 0)]) for z in constrained_zones])
A_arr = np.zeros((T*Z, ZA*T))  
# sa_ind = 0
za_ind = 0
actions = manhattan_game.action_dict.values()
for t in range(T):
    z_ind = 0
    for z in constrained_zones:
        for a_ind in range(len(manhattan_game.action_dict[(z, 0)])):
            A_arr[t*Z+z_ind, a_ind+za_ind] = 1
        za_ind += len(manhattan_game.action_dict[(z, 0)])
        z_ind += 1 
two_norm_A = np.linalg.norm(A_arr,2)
step_size = alpha/2/(two_norm_A**2)
print(f'norm of A is {two_norm_A}, step size is {step_size}')
# step_size = 0.12 #  0.0001
# set initial toll value      
tau = {}
for s in constrained_zones:
    tau.update({((s, 0), t): 0 for t in range(T)})

# bookkeeping stats
average_tau = {z: 0 for z in tau.keys()}
avg_distribution = [{sa: 0 for sa in y_res[-1][t].keys()} 
                    for t in range(T)]
avg_violation = []
social_cost = []

tau_hist = []
violation_hist = []
# define dual ascent approximate gradient update.
def approx_gradient(game, cur_tau, epsilon, k): # this eps comes from inexact_pga
    game.update_tolls(cur_tau)
    # solve first game
    approx_y, obj_hist = fw.FW_dict(game, max_error, max_iterations, 
                                    initial_density, verbose=False)  
    # average tau
    for z in cur_tau.keys():
        average_tau[z] = (average_tau[z]*k+cur_tau[z])/(k+1) 
    # average distribution
    distribution_diff = 0
    for t in range(T):
        for sa in approx_y[-1][t].keys():
            avg_distribution[t][sa] = (
                avg_distribution[t][sa]*k+approx_y[-1][t][sa])/(k+1)
            distribution_diff += abs(avg_distribution[t][sa] - 
                                     approx_y[-1][t][sa])
    # print(f' {k}: distribution difference {round(distribution_diff,2)}')
    gradient, violation = manhattan_game.get_constrained_gradient(
        approx_y[-1], return_violation=True)
    _, avg_violation_k = manhattan_game.get_constrained_gradient(
        avg_distribution, return_violation=True)
    avg_violation.append(avg_violation_k)
    print(f'{k}: average violation {round(avg_violation[-1],2)}')  
    social_cost.append(game.get_social_cost(avg_distribution))
    tau_arr = np.array(list(cur_tau.values()))
    tau_norm = np.linalg.norm(tau_arr, 2)
    print(f'{k}: tau norm {tau_norm}')
    return gradient
    
# epsilons = [epsilon for i in range(max_iteration)]
tau_hist, gradient_hist = pga.inexact_pga(manhattan_game, tau, approx_gradient, step_size, 
                                          max_iterations, epsilons = [max_error]*100000, 
                                          verbose = False)

# find average tau value
tau_values = []
average_tau = {z: 0 for z in tau_hist[-1].keys()}
tau_ind = 0
for tau_ind in range(len(tau_hist)):
    for z in tau.keys():
        average_tau[z] = (average_tau[z]*tau_ind+tau_hist[tau_ind][z])/(tau_ind+1) 
    tau_arr = np.array(list(average_tau.values())) 
    tau_values.append(np.linalg.norm(tau_arr, 2))
# find average constraint violation
pt.objective(tau_values, optimal_value = None, alg_name='Average Toll Value')   

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