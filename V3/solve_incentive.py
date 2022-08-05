# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:23:34 2021

Perform inexact dual ascent

@author: Sarah Li
"""
import numpy as np
import algorithm.FW as fw
import algorithm.inexact_projected_gradient_ascent as pga
import models.mdpcg as mdpcg
import util.plot_lib as pt
import util.utilities as ut

# optimal constrained solution ~= 416971.63808424
time_steps = 20
game = mdpcg.quad_game(time_steps)
p0 = np.zeros((game.G.number_of_nodes()));
# make all drivers start from residential areas 6 of them
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

# constraint matrix generation
constrained_times = [t for t in range(3, time_steps)]
constrained_state = 6  # Belltown y(t,y,:) geq constrained_value
constrained_value = 10
A_tensor = np.zeros((len(constrained_times), game.States, game.Actions, 
                     game.Time))
for t in constrained_times:
   A_tensor[t-constrained_times[0], constrained_state, :, t] = 1
A_array = np.reshape(A_tensor, 
                     (len(constrained_times), game.Time*game.States*game.Actions))

alpha = 1. # strong convexity factor of original functions
two_norm_A = np.linalg.norm(A_array,2)
step_size = alpha/2/(two_norm_A**2)
print(f'norm of A is {two_norm_A}, step size is {step_size}')

distribution_history = [np.zeros((game.States, game.Actions, game.Time))]

# define dual ascent approximate gradient update.
def approx_gradient(tau, epsilon):
    x0 = np.zeros((game.States, game.Actions, game.Time))
    t_offset = constrained_times[0]
    game.reset_toll()
    for t in constrained_times:
        game.tolls += tau[t-t_offset] * A_tensor[t-t_offset, :, :, :]
        
    approx_y = fw.FW(x0, p0, game.P, game.evaluate_cost, maxError=epsilon, 
                     returnHist=False)
    gradient = [constrained_value - np.sum(approx_y[constrained_state, :, t])
                for t in constrained_times]
    distribution_history.append(approx_y)
    return np.array(gradient)

tau_0 = np.ones(len(constrained_times)) * 1200
# tau_0 = np.zeros((len(constrained_times)))
max_iteration = 200
epsilon_list = [1 for i in range(max_iteration)]
tau_hist, gradient_hist = pga.inexact_pga(tau_0, approx_gradient, step_size, 
                                      max_iteration = max_iteration, 
                                      epsilons = epsilon_list,
                                      verbose = True)
average_tau = ut.cumulative_average(tau_hist)
average_y = ut.cumulative_average(distribution_history)

average_constraint_violations= []
for distribution in average_y:
    threshold = [
        constrained_value - np.sum(distribution[constrained_state, :, t]) 
        for t in constrained_times]
    violation = [v for v in threshold if v > 0]
    constraint_violation.append(np.linalg.norm(violation, 2))
    
# pt.latex_format()
# optimal dual - from constrained cvxpy solution
optimal_dual = np.array([
    1289.24191851, 1324.93257885, 1294.36782219, 1318.40797383, 1296.89022284,
    1316.11781582, 1298.90883945, 1314.31790437, 1300.52016307, 1312.87986734,
    1301.81530676, 1311.806991,   1303.01814526, 1313.76323338, 1307.12977287,
    1341.39688255, 1343.22769495])

pt.objective([np.linalg.norm(x, 2) for x in tau_hist], 
             np.linalg.norm(optimal_dual, 2), 'Incentive convergence')
plt.ylabel(r"$\frac{\| \tau^k \| - \| \tau^\star \|}{\| \tau^\star\|}$")
pt.objective(constraint_violation, 1e-5, 'Constraint violation')
plt.ylabel(r"$\|Ay^k - b\|$")