# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:42:08 2022

@author: Sarah Li
"""
import numpy as np
import algorithm.FW as fw
import models.queued_mdp_game as queued_game
import models.taxi_dynamics.manhattan_transition as m_dynamics
import models.taxi_dynamics.visualization as visual
import util.plot_lib as pt
import matplotlib.pyplot as plt
import matplotlib as mpl
import models.taxi_dynamics.manhattan_neighbors as m_neighbors

mpl.rc('text', usetex=False)
mass = 10000
# for debugging
# np.random.seed(49952574)
# print(f' current seed is {np.random.get_state()[1][0]}')
# np.random.seed(3239535799)
manhattan_game = queued_game.queue_game(mass, 0.01, uniform_density=True, 
                                        flat=False)
T = len(manhattan_game.forward_P)
initial_density = manhattan_game.get_density()
y_res, obj_hist = fw.FW_dict(manhattan_game, 
                             max_error=1000, max_iterations=1e3)
print(f'FW solved objective = {obj_hist[-1]}')
z_density = manhattan_game.get_zone_densities(y_res[-1], False)
avg_density = {}
t_density = {}
    
constraint_violation= {}
min_density = 1
max_density = 1
constrained_value = 350
# determine min/max density levels
for t in range(T): 
    min_density = min(list(z_density[t].values()) + [min_density])
    max_density = max(list(z_density[t].values()) + [max_density])
    
print(f'minimum density = {min_density}')
print(f'maximum density = {max_density}')

# determine each state action's constraint violation
violation_density = {}
for z in z_density[0].keys():
    t_density[z] = [z_density[t][z] for t in range(T)]
    if type(z) == tuple:
        if z[1] == 0:
            avg_density[z[0]] = sum(t_density[z])/T
    else:
        avg_density[z] = sum(t_density[z])/T
    threshold = [z_density[t][z] - constrained_value for t in range(T)]
    
    violations = [v for v in threshold if v > 0]
    violation =  sum([v for v in threshold if v > 0]) / T
    if violation > 0:
        constraint_violation[z] = violation
        print(f'zone {z} violates constraint <{ constrained_value} '
              f'with {violation}')
        violation_density[z] = [z_density[t][z] for t in range(T) ]
        if type(z) == tuple:
            z_density[t][z[0]] = z_density[t][z]
            constraint_violation[z[0]] = violation
            constraint_violation.pop(z)

norm = mpl.colors.Normalize(vmin=(min_density), vmax=(max_density))
color_map = plt.get_cmap('coolwarm') # Spectral
bar_colors = []
for violation in constraint_violation.values():
    R,G,B,A = color_map(norm(violation + constrained_value))
    bar_colors.append([R,G,B])   
bar_labels = []  
for z in constraint_violation.keys():
    found = False
    for zone_name, ind in m_neighbors.ZONE_IND.items():
        if found:
            break
        if ind == z:
            bar_labels.append(zone_name)
            found = True
        # elif type(z) == tuple and z[0] == ind:
        #     bar_labels.append(zone_name)
        #     found = True
            
fig_width = 5.3 * 2
f = plt.figure(figsize=(fig_width,8))
ax_bar = f.add_subplot(2,2,1)
bar_range = np.arange(1, len(constraint_violation) + 1, 1)
seq = [i for i in range(len(constraint_violation))] #
violations = []
for v in constraint_violation.values(): 
    violations.append(v)
loc_ind = 0
for bar_ind in seq:
    ax_bar.bar(loc_ind, violations[bar_ind] + constrained_value, 
            width = 0.8,  #color=bar_colors[bar_ind], 
            label=bar_labels[bar_ind])
    loc_ind +=1
ax_bar.set_ylim([constrained_value, max_density])
ax_bar.xaxis.set_visible(False)
plt.legend(fontsize=13)


ax_time = f.add_subplot(2, 2, 3)
ax_time.plot(constrained_value * np.ones(T), 
          linewidth = 6, alpha = 0.5, color=[0,0,0])
lines = []
for line in violation_density.values(): 
    lines.append(line)
for line_ind in seq:
    plt.plot(lines[line_ind], linewidth=3, # color=bar_colors[line_ind], 
              label=bar_labels[line_ind])
plt.xlabel(r"Time",fontsize=13)
plt.grid()
plt.legend(fontsize=13)

ax_map = f.add_subplot(1,2,2)
visual.draw_borough(ax_map, avg_density, 'Manhattan', 'average', color_map, norm)
ax_map.xaxis.set_visible(False)
ax_map.yaxis.set_visible(False)
plt.show()


# violation_states = list(constraint_violation.keys())
# visual.animate_combo('queued_density_unconstrained.mp4', violation_states, t_density, 
#                       bar_labels, T, 'Manhattan', color_map, norm)