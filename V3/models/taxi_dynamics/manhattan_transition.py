# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:31:21 2021

@author: Sarah Li
"""
import models.taxi_dynamics.manhattan_neighbors as manhattan
import numpy as np
import pandas as pd


def random_demand_generation(T, S):
    P_pick_up = np.zeros((T,S, S))
    demand_rate = []
    
    np.random.seed(10)
    for s in range(S):
        demand_rate.append(np.random.randint(1e1, 2e2))
        for t in range(T):
            destinations = np.random.rand(S)
            P_pick_up[t, :, s] = destinations / np.sum(destinations)
        
    return P_pick_up, demand_rate

# Input file name and number of time partitions, returns numpy array of arrays
# Transition kernel: ...\\mdpcg\\V3\\transition_kernel.csv
# Trip Count matrix: ...\\mdpcg\\V3\\count_kernel.csv
def extract_kernel(file, Timesteps, States):
    """ Extract transition kernel per time step from file.
    
    Args:
        file: name of file
        Timesteps: number of time steps.
        States: number of states.
    Returns:
        kernel_list: a list of Timesteps length, each element is an array
          with (States, States) shape, where the [i,j]th component is the
          probability of transitioning from state i to state j.
    """
    kernel_array = pd.read_csv(file, header=0).values
    print(kernel_array.shape)
    kernel_list = [kernel_array[:, t*States: (t+1) * (States)].T 
                   for t in range(Timesteps)]
    return kernel_list
    


def uniform_initial_distribution(M):
    """ Return a uniform density array of drivers in Manhattan states. """
    state_num = len(manhattan.STATE_NEIGHBORS)
    p0 = np.ones(state_num) * M / state_num
    return p0

def random_distribution(M, constraint_val):
    """ Return a random density array of drivers in Manhattan states. """
    state_num = len(manhattan.STATE_NEIGHBORS)
    p0 = np.random.rand(state_num)
    p0 = p0 / sum(p0) * M
    constraint_violated = True
    while constraint_violated:
        constraint_violated = False
        for s in range(len(p0)):
            if p0[s] > constraint_val:
                constraint_violated = True
                p0 += (p0[s] - constraint_val) / (state_num - 1)
                p0[s] = constraint_val
    return p0


def transition_kernel(T, epsilon):
    """ Return a  4 dimensional transition kernel for Manhattan's MDP dynamics.
    
    Args: 
        T: total number of time steps within the MDP
        epsilon: the probability of not getting to neighbor state
    Returns:
        P: [T] x [S] x [S] x [A] transition kernel as ndarray.
        P_{ts'sa} is the probability of transitioning to s' from (s,a) at t.
    """
    S = len(manhattan.STATE_NEIGHBORS)
    A = manhattan.most_neighbors(manhattan.STATE_NEIGHBORS)
    # true action is A + 1: last action is reserved for picking up passengers.
    P_t = np.zeros((S, S, A + 1))  # kernel per time step. 
    
    for state, neighbors in manhattan.STATE_NEIGHBORS.items():
        N_n = len(neighbors) # number of neighbors
        # probability of arriving at correct neighbor
        p_target = 1 - N_n / (N_n -1) * epsilon
        # probability of arriving at another neighbor
        p_other_neighbor = epsilon/(N_n - 1) 
        
        action_ind = -1
        while action_ind < A - 1:
            action_ind += 1
            neighbor = neighbors[-1]
            if action_ind < N_n:
                neighbor = neighbors[action_ind]
            # action goes to correct neighbor
            P_t[neighbor, state, action_ind] = p_target
            # action may take player to other neighbors
            for other_n in neighbors:
                P_t[other_n, state, action_ind] += p_other_neighbor
                
    P = np.zeros((T, S, S, A + 1))
    for t in range(T):
        P[t, :, :, :] = P_t
    return P
            
def test_transition_kernel(P):
    (T, S, _, A) = P.shape
    for t in range(T):
        for a in range(A-1):
            M = P[t, :, :, a]
            for s in range(S):
                col_sum = np.sum(M[:,s])
                # column stochasticity
                np.testing.assert_approx_equal(col_sum, 1, 4)