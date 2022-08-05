# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:20:58 2021

Tolled version of Manhattan game.
@author: Sarah Li
"""
import numpy as np
import algorithm.FW as fw
import algorithm.inexact_projected_gradient_ascent as pga
import models.mdpcg as mdpcg
import models.taxi_dynamics.manhattan_transition as m_dynamics
import models.taxi_dynamics.manhattan_neighbors as m_neighbors
import models.taxi_dynamics.visualization as visual
import util.plot_lib as pt
import util.utilities as ut
import matplotlib.pyplot as plt
import matplotlib as mpl


T = 15 # number of time steps of MDP
epsilon = 10000 # error in approximate solution of drivers learning
epsilon_list = [5e3]  # [1e5, 2e4, 2e3, 1e3]
borough = 'Manhattan' # borough of interest
fleet_size = 10000 # game population size
constrained_value = 400 # maximum driver density per state
max_iteration = 2000 # number of iterations of dual ascent

# game definition with initial distribution
manhattan_game = mdpcg.quad_game(T, manhattan=True)
initial_distribution = m_dynamics.uniform_initial_distribution(fleet_size)
initial_distributions = [
    m_dynamics.random_distribution(fleet_size, constrained_value) for i in range(10)]
S = manhattan_game.States
A = manhattan_game.Actions

# create constraint tensor
A_tensor = np.zeros((T*S, S, A, T))  
for t in range(T):
    for s in range(S):
        A_tensor[t*S+s, s, :, t] = 1
A_array = np.reshape(A_tensor, (T*S, S*A*T))      

# strong convexity factor of original functions
alpha = np.min(manhattan_game.R)
print(f'convexity factor is {alpha}')

two_norm_A = np.linalg.norm(A_array,2)
step_size = 0.001
step_size = alpha/2/(two_norm_A**2)
print(f'norm of A is {two_norm_A}, step size is {step_size}')

average_tau = []
average_y = []
true_constraint_violation = []
init_dist  = m_dynamics.uniform_initial_distribution(fleet_size)
distribution_history = []
# define dual ascent approximate gradient update.
def approx_gradient(tau, epsilon): # this eps comes from inexact_pga
    x0 = np.zeros((S, A, T));   
    manhattan_game.reset_toll()
    for t in range(T):
        manhattan_game.tolls += sum([
            tau[t, s] * A_tensor[t*S + s, :, :, :] for s in range(S)])
    # solve first game
    approx_y = fw.FW(x0, init_dist, manhattan_game.P, 
                        manhattan_game.evaluate_cost, maxIterations = 1e6,
                        maxError=epsilon, returnHist=False)
    distribution_history.append(approx_y)

    pop_distribution = np.sum(approx_y, axis=1) # size TxS
    constraint_arr = np.ones(pop_distribution.shape) * constrained_value
    gradient = pop_distribution - constraint_arr # gradient has size T x S

    # for t in range(T):
    #     gradient.append([
    #         np.sum(approx_y[s,  :, t]) - constrained_value 
    #         for s in range(S)])
    #     violation += sum([max([0, g]) for g in gradient[-1]])
        
    return gradient.T
    
tau_0 = np.zeros((T,S))
epsilons = [epsilon for i in range(max_iteration)]
tau_hist, gradient_hist = pga.inexact_pga(tau_0, approx_gradient, 
                                            step_size,
                                            max_iteration = max_iteration, 
                                            epsilons=epsilons, 
                                            verbose = True)
for grad in gradient_hist:
    grad[grad< 0] = 0
    true_constraint_violation.append(np.linalg.norm(grad, 2))

# average population results
average_tau.append(ut.cumulative_average(tau_hist))
average_tau[-1].pop(0)
average_y.append(ut.cumulative_average(distribution_history))

#-------------dual ascent and constraint violation convergence --------------------
constraint_violation= []
toll_values = []
Iterations = len(average_y)
for ind in range(Iterations):
    constraint_violation.append([])
    toll_values.append([np.linalg.norm(x, 2) for x in average_tau[ind]])
    for y in average_y[ind]:
        threshold = []
        for s in range(S):
            threshold = threshold + [np.sum(y[s, :, t]) - constrained_value 
                                     for t in range(T)]    
        violation = np.array([v for v in threshold if v > 0])
        constraint_violation[-1].append(np.linalg.norm(violation, 2))

#------- saving info for comparison with the fast gradient method ------------#        
tau_norm_grad = [np.linalg.norm(tau, 2) for tau in tau_hist]
tau_avg_grad = [np.linalg.norm(x, 2) for x in average_tau[-1]]
constraint_violation_avg_grad = constraint_violation[-1]
constraint_violation_grad = true_constraint_violation

#--------------epsilon vs toll and constraint violation-------------#
iteration_line = np.linspace(1, len(average_y[-1]),len(average_y[-1]))
if len(epsilon_list) > 1:
    fig_width = 5.3 * 2
    epsilon_plot = plt.figure(figsize=(fig_width,8))
    toll_plot = epsilon_plot.add_subplot(2,1,1)
    plt.plot(epsilon_list, [toll_values[ind][-1] for ind in range(Iterations)], 
            linewidth=3)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.setp(toll_plot.get_xticklabels(), visible=False)
    epsilon_plot.add_subplot(2, 1, 2, sharex =toll_plot )
    plt.plot(epsilon_list, [constraint_violation[ind][-1] for ind in range(Iterations)], 
            linewidth=3)
    plt.xscale('log')
    plt.xlabel('$\epsilon$',fontsize=12)
    plt.yscale('log')
    plt.grid()
    plt.subplots_adjust(hspace=.0)


plt.figure()
print('fig 1 = toll norm as function of designer iteration')
if len(epsilon_list) > 1:
    plt.plot(epsilon_list, [toll_values[ind][-1] for ind in range(Iterations)], 
            linewidth=3)
else:
    plt.plot(toll_values[0], linewidth=3)
    toll_value_grad = toll_values[0] # save for fast grad comparison
plt.xscale('log')
plt.xlabel('$\epsilon$',fontsize=12)
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('grad_res/toll_norm_as_function_of_designer_iteration.png')

plt.figure()
print('fig 2 = constraint_violation as function of designer iteration')
if len(epsilon_list) == 1:
    plt.plot(true_constraint_violation, linewidth=3,label='true $\hat{y}^k$ violation')
    plt.plot(constraint_violation[ind], linewidth=3,label='average $\hat{y}^k$ violation')
    plt.xlabel('$k$',fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
else:
    plt.plot(epsilon_list, [constraint_violation[ind] for ind in range(Iterations)], 
             linewidth=3)
    plt.xlabel('$\epsilon$',fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('grad_res/constraint_violation_average_vs_true.png')

#-----------average tolling values for last approximation ------------
plt.figure()
print('fig 3 = toll norm as function of designer iteration')
plt.plot(iteration_line, toll_values[-1], linewidth=3, 
          label='total toll value')
plt.plot(iteration_line, constraint_violation[-1],
         label = 'total constraint violation ', linewidth=3)
plt.legend()
plt.xlabel('Iterations')
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('toll_norm_convergence.png')

#-------------tolling system level info --------------------
expected_mdp_cost = []
toll_received = []
for ind in range(len(average_y[-1])):
    # calculate each iteration's driver reward
    cur_cost = manhattan_game.evaluate_social_cost(average_y[-1][ind])
    expected_mdp_cost.append(-np.sum(cur_cost)) # negative cost = reward   
    # calculate the total toll received by system
    state_time_density = np.sum(average_y[-1][ind], axis=1)
    toll_received.append(np.sum(
        np.multiply(state_time_density, average_tau[-1][ind].T)))
    # print(np.sum(cur_cost))
print('fig 3 = Total profit from toll as function of designer iteration')
fig = plt.figure()
pt.objective(toll_received, None, 'Total toll profit')
plt.tight_layout()
plt.savefig('grad_res/total_profit.png')

print('fig 4 = Average driver profit as function of designer iteration')
pt.objective(expected_mdp_cost, expected_mdp_cost[0], 'Social Cost for driver')  
plt.tight_layout()
plt.savefig('grad_res/social_driver_profit.png')

avg_cost = 10000 # average cost of the game

#-----------visualize optimal distribution -----------------
# visual.plot_borough_progress(borough, average_y[-1][-1], [0, int(T/2), T-1])

state_density = {}
state_ind = m_neighbors.zone_to_state(m_neighbors.zone_neighbors)
zone_ind = {y:x for x, y in state_ind.items()}
evaluated_y = average_y[-1][-1]
evaluated_tau = average_tau[-1][-1]
for s in range(manhattan_game.States):
    threshold = [
        np.sum(evaluated_y[s, :, t]) - constrained_value for t in range(T)]    
    state_density_avg = np.sum([evaluated_y[s, :, t] for t in range(T)])/T
    state_density[zone_ind[s]] = state_density_avg
    
# calculate color bars
min_density = 1 # 26.392308291054466
max_density = 510 # 417.1387978157465
norm = mpl.colors.Normalize(vmin=min_density, vmax= max_density)
color_map = plt.get_cmap('coolwarm')
bar_labels = ['Financial District North', 'Midtown Center', 'World Trade Center']  
# bar_labels = ["Midtown Center", 'Upper East Side N', 'Upper East Side S']
bar_colors = []

tolled_states =  [87, 161, 261] # [161, 236, 237]#
for z in tolled_states:
    R,G,B,A = color_map(norm(state_density[z] + constrained_value))
    bar_colors.append([R,G,B])   
 

# calculate average constraint violation
violation_density = {}
tolls = {}
for z in tolled_states:  
    s_ind = state_ind[z]
    violation_density[z] = [
        np.sum(evaluated_y[s_ind, :, t]) for t in range(T)]
    tolls[z] = evaluated_tau[:, s_ind]

# plot the time averaged constrained density, time dependent tolls and density
print('fig 5 = Composed figure with toll, violation, manhattan map')
fig_width = 5.3 * 2
f = plt.figure(figsize=(fig_width,8))
seq = [0, 2,  1]
ax_toll_val = f.add_subplot(2,2,1) # (2,2,2)
toll_time_vary = []
for line in tolls.values(): 
    toll_time_vary.append(line)
for line_ind in seq:
    ax_toll_val.plot(toll_time_vary[line_ind], linewidth=3, 
                     label=bar_labels[line_ind])
plt.setp(ax_toll_val.get_xticklabels(), visible=False)
plt.grid()

ax_constrained_density = f.add_subplot(2,2,3,sharex=ax_toll_val) # (2, 2, 4)
plt.plot(constrained_value*np.ones(T), linewidth = 6, alpha = 0.3, color=[0,0,0])
lines = []
for line in violation_density.values(): 
    lines.append(line)
for line_ind in seq:
    ax_constrained_density.plot(lines[line_ind], linewidth=3, 
                                label=bar_labels[line_ind])
plt.xlabel(r"Time",fontsize=13)

plt.grid()
plt.legend(fontsize=13)

ax = f.add_subplot(1,2,2)#(1, 2, 1)
visual.draw_borough(ax, state_density, borough, 'average', color_map, norm)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.show()
plt.tight_layout()
plt.savefig('grad_res/composed_manhattan.png')

#------------------- constraint violation ----------------------
print('fig 6 = Total constraint violation vs gradient descent iteration')
plt.figure()
plt.title('Total constraint violation vs gradient descent iteration')
plt.plot(true_constraint_violation, linewidth = 3,)
plt.xscale('linear')
plt.xlabel('$k$',fontsize=12)
plt.yscale('linear')
plt.grid()
plt.show()
plt.savefig('grad_res/total_constriant_violation.png')

# density_t = {}
# for s in range(manhattan_game.States):
#     density_t[zone_ind[s]] = [sum(evaluated_y[s,:,t]) for t in range(T) ]

# visual.animate_combo('density_tolled.mp4', tolled_states, density_t, 
#                      bar_labels, T, borough, color_map, norm, toll_time_vary)





plt.figure()
print('fig 2 = constraint_violation as function of designer iteration')

plt.plot(constraint_violation_grad, '-.', linewidth=6,
          label='standard grad. desc.(true)',color='C0', alpha=0.5)
plt.plot(constraint_violation_avg_grad, linewidth=3,
          label='standard grad. desc. (avg)',color='C0')

plt.plot(constraint_violation_fast, '-.', linewidth=6,
          label='fast grad. desc. (true)',color='C1', alpha = 0.5)
plt.plot(constraint_violation_avg_fast, linewidth=3,label='fast grad. desc. (avg)',color='C1')
plt.xlabel('$k$',fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.grid()
plt.tight_layout()
plt.savefig('comparison/constraint_violation.png')


#-----------average tolling values for last approximation ------------
plt.figure()
print('fig 3 = toll norm as function of designer iteration')
plt.plot(tau_norm_grad, linewidth=6, 
          label='standard grad. desc. (true)', color='C0', alpha = 0.5)
plt.plot(tau_avg_grad, linewidth=3,
          label='standard grad. desc. (avg)',color='C0')

plt.plot(tau_norm_fast, '-.', linewidth=6,
          label='fast grad. desc. (true)',color='C1', alpha = 0.5)
plt.plot(tau_avg_fast, linewidth=3,label='fast grad. desc. (avg)',color='C1')
plt.legend()
plt.xlabel('Iterations')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('toll_norm_convergence.png')

#-------------tolling system level info --------------------

print('fig 3 = Total profit from toll as function of designer iteration')
fig = plt.figure(figsize=(6,3))
plt.plot([((x - expected_mdp_cost[0])/expected_mdp_cost[0]) for x in expected_mdp_cost], 
     linewidth=3, 
     label='standard grad. desc.')
plt.plot([((x - expected_mdp_cost_fast[0])/expected_mdp_cost_fast[0]) 
          for x in expected_mdp_cost_fast], 
     linewidth=3, 
     label='fast grad. desc.', color='C1')
plt.tight_layout()
plt.legend()
plt.xlabel('Iterations')
# plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('compare/social_cost.pdf')

fig = plt.figure()
plt.plot([((x - expected_mdp_cost_fast[0])/expected_mdp_cost_fast[0]) 
          for x in expected_mdp_cost_fast], 
     linewidth=3, 
     label='fast grad. desc.', color='C1')
plt.tight_layout()
plt.legend()
plt.xlabel('Iterations')
# plt.yscale('log')
plt.grid()
plt.savefig('grad_res/social_cost.png')
