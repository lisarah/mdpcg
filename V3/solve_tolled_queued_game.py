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
import pickle

# epsilon_list = [5e3]  # [1e5, 2e4, 2e3, 1e3]
borough = 'Manhattan' # borough of interest
mass = 10000 # game population size
constrained_value = 350 # 350  for flat # maximum driver density per state
max_errors = [5000]#[10000, 5000, 1000, 500, 100]
max_iterations = 10 # number of iterations of dual ascent
toll_queues = False
save_last_toll_results = True
save_plots = True
# game definition with initial distribution
# print(f'cur seed {np.random.get_state[1][0]}')
# np.random.seed(3239535799)
manhattan_game = queued_game.queue_game(mass, 0.01, uniform_density=True,
                                        flat=False) 

initial_density = manhattan_game.get_density()
y_res, obj_hist = fw.FW_dict(manhattan_game, max_errors[-1], max_iterations)

# set constraints 
constrained_zones = [161, 162, 236, 237] #[161,43,68,79,231,236,237,114]# manhattan_game.z_list
manhattan_game.set_constraints(constrained_zones, constrained_value)
grad, violation_norm = manhattan_game.get_constrained_gradient(
    y_res[-1], return_violation = True)
print(f'before tolling: violation_density = {violation_norm}')


# step size calculations. Step size =  alpha / 2 / |A|^2_2
# alpha is alpha strong convexity, |A|^2_2 is the two norm of constraint matrix
# and the constraint norm
alpha = manhattan_game.get_strong_convexity()
print(f'convexity factor is {alpha}')
T = len(manhattan_game.forward_P)
Z = len(constrained_zones)
ZA = sum([len(manhattan_game.action_dict[(z, 0)]) for z in constrained_zones])
A_arr = np.zeros((T*Z, ZA*T))  
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


# Initialize toll value at tolled states
tau = {}
for s in constrained_zones:
    tau.update({((s, 0), t): 0 for t in range(T)})

# error vs accuracy stats
avg_tolls = {err: None for err in max_errors}
avg_violations = {err: None for err in max_errors}

# loop for running inexact pga at different error accuracies
for err in max_errors:
    print('')
    print(f'------ running error {err} ---------')

    # bookkeeping stats
    avg_distribution = [{sa: 0 for sa in y_res[-1][t].keys()} 
                        for t in range(T)]
    avg_violation = []
    social_cost = []
    # define dual ascent approximate gradient update.
    # input approx_err comes from inexact_pga
    def approx_gradient(game, cur_tau, approx_err, k): 
        game.update_tolls(cur_tau)
        # solve first game
        approx_y, obj_hist = fw.FW_dict(game, approx_err, max_iterations, 
                                        initial_density, verbose=False)  
        for t in range(T):
            for sa in approx_y[-1][t].keys():
                avg_distribution[t][sa] = (
                    avg_distribution[t][sa]*k+approx_y[-1][t][sa])/(k+1)
                
        gradient, violation = manhattan_game.get_constrained_gradient(
            approx_y[-1], return_violation=True)
        _, avg_violation_k = manhattan_game.get_constrained_gradient(
            avg_distribution, return_violation=True)
        avg_violation.append(avg_violation_k)
        social_cost.append(game.get_social_cost(avg_distribution))
        tau_norm = np.linalg.norm(np.array(list(cur_tau.values())), 2)
        print(f'\r {k}: average violation {round(avg_violation[-1],2)}, '
              f'tau norm {tau_norm}   ', end ='')  
        return gradient
    
    tau_hist, gradient_hist = pga.inexact_pga(manhattan_game, tau,
                                              approx_gradient, step_size,
                                              max_iterations, 
                                              epsilons=[err]*100000, 
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
    plot_title = f'Average Toll Value error = {err}'
    pt.objective(tau_values, optimal_value = None, alg_name=plot_title)   
    
    #%% average tolling values for last approximation %%#
    plt.figure()
    # print('fig 3 = toll norm as function of designer iteration')
    plt.title(f'at error = {err}')
    plt.plot(tau_values, linewidth=3, label='total toll value')
    plt.plot(avg_violation, label = 'total constraint violation', linewidth=3)
    plt.legend()
    plt.xlabel('Iterations')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'toll_norm_convergence_{err}.png')
    
    
    # social cost plot
    pt.objective(social_cost, social_cost[0], 'Social Cost for driver')  
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'grad_res/social_driver_profit_{err}.png')
    
    avg_tolls[err] = tau_values[-1]
    avg_violations[err] = avg_violation[-1]
    
    
#%% plots %%#
# plot errors vs last average toll value and last violation violation
x_axis_labels = [e /200839.1820886145 for e in max_errors]
if len(max_errors) > 1:
    fig_width = 5.3 * 2
    epsilon_plot = plt.figure(figsize=(fig_width,8))
    toll_plot = epsilon_plot.add_subplot(2,1,1)
    plt.plot(x_axis_labels, avg_tolls.values(), linewidth=5)
    plt.ylabel('$\|| \hat{ τ}^k\||_2$', fontsize=18)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.setp(toll_plot.get_xticklabels(), visible=False, fontsize=18)
    plt.setp(toll_plot.get_yticklabels(), fontsize=18)
    plt.setp(toll_plot.get_yticklabels(minor=True), fontsize=18)
    violation_plot = epsilon_plot.add_subplot(2, 1, 2, sharex=toll_plot) #
    plt.plot(x_axis_labels, avg_violations.values(), linewidth=5)
    plt.ylabel('$||A\hat{y}^k- b||_2$', fontsize=18)
    plt.xscale('log')
    plt.xlabel('$\epsilon$',fontsize=18)
    plt.yscale('log')
    plt.setp(violation_plot.get_xticklabels(), fontsize=18)
    plt.setp(violation_plot.get_yticklabels(), fontsize=18)
    plt.setp(violation_plot.get_yticklabels(minor=True), fontsize=18)
    
    plt.grid()
    plt.subplots_adjust(hspace=.05)    
    plt.show()

if save_last_toll_results:
    open_file = open('grad_res/err_5000_toll_results.pickle', 'wb')
    err_results = {'max_error': 5000,
                   'tau_hist': tau_hist,
                   'gradient_hist':gradient_hist,
                   'avg_violation': avg_violation,
                   'step_size': step_size,
                   'social_cost': social_cost,
                   'last_average_density': avg_distribution}
    pickle.dump(err_results, open_file)
    open_file.close()

# plot the summary plot of the tolled average results
# get summary friendly densities
z_density = manhattan_game.get_zone_densities(avg_distribution, False)
violation_density, constraint_violation = manhattan_game.get_violation_subset(
    z_density, constrained_zones, constrained_value)
avg_density = manhattan_game.get_average_density(z_density)
# derive time varying tolls of the constrained states:
avg_toll_time = {v_key: [average_tau[((v_key, 0), t)] for t in range(T)]
                 for v_key in violation_density.keys()}

visual.summary_plot(z_density, constraint_violation, violation_density, 
                    avg_density, constrained_value, avg_toll_time)