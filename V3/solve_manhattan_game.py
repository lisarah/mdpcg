# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:07:53 2021

@author: Sarah Li
"""
import numpy as np
import models.taxi_dynamics.manhattan_neighbors as manhattan

T = 15
# the last action in P is for the action of trying to pick up drivers.
P = manhattan.manhattan_transition_kernel(T, 0.1)
manhattan.test_manhattan_transition_kernel(P)
# random rider demand generation
T,S, _, A = P.shape
print(f'MDP has time steps={T}, states={S}, actions={A}')

def random_demand_generation(T, S):
    P_pick_up = np.zeros((T,S, S))
    demand_rate = []
    for s in range(S):
        demand_rate.append(np.random.randrange(1e4, 1e5))
        for t in range(T):
            destinations = np.random.rand(S)
            P_pick_up[t, :, s] = destinations / np.sum(destinations)
        
    return P_pick_up, demand_rate

P_pick_up, demand_rate = random_demand_generation(T,S)
P[:,:,:, A-1] += P_pick_up
manhattan.test_manhattan_transition_kernel(P)

