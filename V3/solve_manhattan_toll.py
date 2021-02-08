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


T = 15
epsilon = 1e-1
borough = 'Manhattan'
manhattan_game = mdpcg.quad_game(T, manhattan=True)
initial_distribution = m_dynamics.uniform_initial_distribution(10000)

S = manhattan_game.States
A = manhattan_game.Actions
A_tensor = np.zeros((T*S, S, A, T))  
constrained_value = 250
x_ind = 0
for t in range(T):
    x_ind = t*S
    for s in range(S):
        A_tensor[x_ind, s, :, t] = 1
        x_ind += 1
A_array = np.reshape(A_tensor, (T*S, S*A*T))      

alpha = np.min(manhattan_game.R) # strong convexity factor of original functions
print(f'convexity factor is {alpha}')

two_norm_A = np.linalg.norm(A_array,2)
# step_size = 0.05
step_size = alpha/2/(two_norm_A**2)
print(f'norm of A is {two_norm_A}, step size is {step_size}')

distribution_history = [np.zeros((S, A, T))]

# define dual ascent approximate gradient update.
def approx_gradient(tau, epsilon):
    x0 = np.zeros((S, A, T));   
    manhattan_game.reset_toll()
    for t in range(T):
        x_ind = t*S
        for s in range(S): 
            manhattan_game.tolls += tau[t,s] * A_tensor[x_ind, :, :, :]
            x_ind +=1
        
    approx_y = fw.FW(x0, initial_distribution, manhattan_game.P, 
                     manhattan_game.evaluate_cost, maxIterations = 1000,
                     maxError=epsilon, returnHist=False)
    # print(np.sum(approx_y[:,:,0]))
    gradient = [] # gradient has size T x S
    violation = 0
    for t in range(T):
        gradient.append([])
        for s in range(S):
            violation += max([0, np.sum(approx_y[s, :, t]) - constrained_value])
            gradient[-1].append(np.sum(approx_y[s, :, t]) - constrained_value)
    print (f'total constraint violation = {violation}')
    distribution_history.append(approx_y)
    return np.asarray(gradient)

tau_0 = np.zeros((T,S))
# tau_0 = np.zeros((len(constrained_times)))
max_iteration = 200
epsilon_list = [15000 for i in range(max_iteration)]
tau_hist, gradient_hist = pga.inexact_pga(tau_0, approx_gradient, step_size, 
                                      max_iteration = max_iteration, 
                                      epsilons=epsilon_list, verbose = True)
  
# average_tau = ut.cumulative_average(tau_hist)
# average_y = ut.cumulative_average(distribution_history)

# constraint_violation= []
# check = 0
# for distribution in average_y:
#     check += 1
#     threshold = []
#     for s in range(S):
#         threshold = threshold + [np.sum(distribution[s, :, t]) 
#                                  - constrained_value for t in range(T)]
#     # if check < 2:
#     #     print(threshold)
        
#     violation = np.array([v for v in threshold if v > 0])
#     constraint_violation.append(np.linalg.norm(violation, 2))


# pt.objective([np.linalg.norm(x, 2) for x in average_tau], 
#              None, 'Incentive convergence')
# plt.ylabel(r"$\|tau^k \|_2$")
# pt.objective(constraint_violation, None, 'Constraint violation')
# plt.ylabel(r"$\|Ay^k - b\|_2$")

#-----------visualize optimal distribution -----------------

visual.plot_borough_progress(borough, distribution_history[-1], [0, int(T/2), T-1])

constraint_violation= {}
state_density = {}
min_density = 999
max_density = -1
state_ind = m_neighbors.zone_to_state(m_neighbors.zone_neighbors)
zone_ind = {y:x for x, y in state_ind.items()}
for s in range(manhattan_game.States):
    threshold = [np.sum(distribution_history[-2][s, :, t]) - constrained_value for t in range(T)]
        
    violations = [v for v in threshold if v > 0]
    violation =  sum([v for v in threshold if v > 0]) / T
    if violation > 0:
        constraint_violation[zone_ind[s]] = violation
    
    state_density_avg = np.sum([distribution_history[-2][s, :, t] for t in range(T)])/T
    state_density[zone_ind[s]] = state_density_avg
    min_density = min([state_density_avg, min_density])
    max_density = max([state_density_avg, max_density])

violation_density = {}
tolls = {}
tolled_states = [249, 261, 263]
for z in tolled_states:  
    time_density = []
    state_toll = tau_hist[-1][:, state_ind[z]]
    for t in range(T):
        time_density.append(np.sum(distribution_history[-2][state_ind[z], :, t]))

    violation_density[z] = time_density
    tolls[z] = state_toll
norm = mpl.colors.Normalize(vmin=(min_density), vmax=(max_density))
color_map = plt.get_cmap('coolwarm')
bar_colors = []
for z in tolled_states:
    R,G,B,A = color_map(norm(state_density[z] + constrained_value))
    bar_colors.append([R,G,B])   
bar_labels = ['West Village', 'World Trade Center', 'Yorkville West']  

fig_width = 5.3 * 2
f = plt.figure(figsize=(fig_width,8))
seq = [0,2,  1]
f.add_subplot(2,2,2)
toll_time_vary = []
for line in tolls.values(): 
    toll_time_vary.append(line)
for line_ind in seq:
    plt.plot(toll_time_vary[line_ind], linewidth=3, label=bar_labels[line_ind])
plt.xlabel(r"Time",fontsize=13)
plt.yscale('log')
plt.grid()
plt.legend(fontsize=13)


f.add_subplot(2, 2, 4)
plt.plot(250*np.ones(T), linewidth = 6, alpha = 0.5, color=[0,0,0])
lines = []
for line in violation_density.values(): 
    lines.append(line)
for line_ind in seq:
    plt.plot(lines[line_ind], linewidth=3, label=bar_labels[line_ind])
plt.xlabel(r"Time",fontsize=13)
plt.yscale('log')
plt.grid()
plt.legend(fontsize=13)

ax = f.add_subplot(1, 2, 1)
visual.draw_borough(ax, state_density, borough, 'average', color_map, norm)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.show()