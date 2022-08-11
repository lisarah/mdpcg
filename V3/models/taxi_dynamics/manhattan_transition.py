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
def transition_kernel_dict(epsilon, transitions_list):
    """

    Parameters
    ----------
    epsilon : float between (0, 1)
        when choosing to go to neighboring state, probability of not 
        getting there.
    transitions_list : list
        Each element of transition_list is t_i.
        t_i: dict: {z_i: t_{ij}}
            z_i =  origin_zone_ind
            t_ij = dict: {dest: p_{ij}}
                dest = (destination zone ind, time)
                p_{ij} = probability of going to dest from z_i. 
    

    Returns
    -------
    forward_transitions: list, [d_i] for all i \in [T]
        d_i = dict, {state_j: t_ij} for all state_j \in [S]
            t_ij = dict, {a_k: P_ijk} for all a_k \in [A_j]
                P_ijk = tuple: (states_list, transitions_list).
                    states_list = list of states (zone_ind, queue_level) to 
                                  transition to
                    transitions_list = list of transition probabilities 
                                       for these states 
                                       at action a_k from state s_j and time i
                                       
                             
    backward_transitions: list, [d_t] for all t \in [T]
        d_t = dict, {state_j: (sa_list, probability_list)} for all j \in [S]
            sa_list = [(s_i, a_k)] for all i \in [N_j], k \in [A_j]
                s_i = (z_i, q_level)
            probability_list = [P_jikt], 
                P_ijkt is the probability of transitioning to state j by
                taking state-action (s_i, a_k) at time t. 

    """
    max_action = manhattan.most_neighbors(manhattan.zone_neighbors)
    forward_transitions = []
    backward_transitions = []
    state_list = []
    max_queue_level = 8 if len(transitions_list) == 15 else 7
    # todo: this should and can be removed.
    truncate_zones =  [103, 104, 105, 153, 194, 202]
    for z_i in manhattan.zone_neighbors:
        if z_i not in truncate_zones:
            for q_level in range(max_queue_level):
                state_list.append((z_i, q_level))
    action_dict = {s: [] for s in state_list}
    for s in state_list:
        if s[1] == 0: # in the pick up queue - add go to neighbors action
            a_ind = 0
            for neighbor in manhattan.zone_neighbors[s[0]]:
                if neighbor not in truncate_zones:
                    action_dict[s].append(a_ind)
                    a_ind += 1
        # legacy: last action is trying to pick up a rider                
        action_dict[s].append(max_action) # always add max_action
        
    sa_list = []
    for s in state_list:
        sa_list = sa_list + [(s,a) for a in action_dict[s]]                    
    
    
    for t_i in transitions_list:
        forward_transitions.append({s:{a:([],[]) for a in action_dict[s]} 
                                    for s in state_list})

        backward_transitions.append({s: ([], []) for s in state_list})
        
        # Add actions depending on state
        for s in state_list:
            # action for queue_level > 0 is just to drop
            if s[1] > 0:
                forward_transitions[-1][s][max_action][0].append((s[0], s[1] - 1))
                forward_transitions[-1][s][max_action][1].append(1)
            if s[1] < max_queue_level - 1: 
                backward_transitions[-1][s][0].append(((s[0], s[1]+1), max_action))
                backward_transitions[-1][s][1].append(1)
                                                      
            # action for queue == 0 contains going to neighbors
            if s[1] == 0:
                # add actions for going to neighbors
                a_ind = -1
                total_neighbors = len(manhattan.zone_neighbors[s[0]])
                for n_j in manhattan.zone_neighbors[s[0]]:  
                    # insert actions in forward_transitions
                    a_ind += 1
                    n_state = (n_j, 0)
                    forward_transitions[-1][s][a_ind] = ([n_state], [1])
                    # insert actions in backward_transtiions
                    backward_transitions[-1][n_state][0].append((s, a_ind))
                    backward_transitions[-1][n_state][1].append(1)
    
                        
                    if epsilon > 0 and total_neighbors > 1:
                        forward_transitions[-1][s][a_ind][1][-1] += -epsilon
                        backward_transitions[-1][n_state][1][-1] += -epsilon
                        for n_k in manhattan.zone_neighbors[s[0]]:
                            other_neighbor_state = (n_k, 0)
                            if n_k != n_j:
                                forward_transitions[-1][s][a_ind][0].append(
                                   other_neighbor_state)
                                forward_transitions[-1][s][a_ind][1].append(
                                    epsilon/(total_neighbors-1))
                                backward_transitions[-1][other_neighbor_state][0].append(
                                        (s, a_ind))
                                backward_transitions[-1][other_neighbor_state][1].append(
                                    epsilon / (total_neighbors-1) )
                            
                                # if other_neighbor_state == (79, 0):
                                #     print(f' found {o_ind} -> {other_neighbor_state} in neighbor loop 1')                                              
        for z_i, t_ij in t_i.items():
            o_ind = (z_i, 0) # queue level is 0
            if len(t_ij) == 0: # no rides recorded from z_i at time t_i
            # if no rides are recorded, go back to origin with probability 1.
                forward_transitions[-1][o_ind][max_action] = ([o_ind], [1])
                backward_transitions[-1][o_ind][0].append((o_ind, max_action))
                backward_transitions[-1][o_ind][1].append(1)
                
            
            # otherwise for the rides recorded:  
            for dest, p_ij in t_ij.items() :
                # build forward transitions connection
                forward_transitions[-1][o_ind][max_action][0].append(dest)
                forward_transitions[-1][o_ind][max_action][1].append(p_ij)
 
                    
                # build backward transitions:
                # add the first transition from picking up a driver
                backward_transitions[-1][dest][0].append((o_ind, max_action))
                backward_transitions[-1][dest][1].append(p_ij)
                # if dest == (79, 0) and o_ind == (79,1):
                #         print(f' found {o_ind} -> {dest} outside while loop')
                        
                    
            
            
                    
    # print(f'final backward transition value {backward_transitions[-1][(79,0)][1][3]}  ')
    # print(f'origin state-action is {backward_transitions[-1][(79,0)][0][3]}')            
    return forward_transitions, backward_transitions, state_list, action_dict, sa_list

def transition_kernel_dict_flat(epsilon, transitions_list):
    """

    Parameters
    ----------
    epsilon : float between (0, 1)
        when choosing to go to neighboring state, probability of not 
        getting there.
    transitions_list : list
        Each element of transition_list is t_i.
        t_i: dict: {z_i: t_{ij}}
            z_i =  origin_zone_ind
            t_ij = dict: {dest: p_{ij}}
                dest = (destination zone ind, time)
                p_{ij} = probability of going to dest from z_i. 
    

    Returns
    -------
    forward_transitions: list, [d_i] for all i \in [T]
        d_i = dict, {state_j: t_ij} for all state_j \in [S]
            t_ij = dict, {a_k: P_ijk} for all a_k \in [A_j]
                P_ijk = tuple: (states_list, transitions_list).
                    states_list = list of states (zone_ind, queue_level) to 
                                  transition to
                    transitions_list = list of transition probabilities 
                                       for these states 
                                       at action a_k from state s_j and time i
                                       
                             
    backward_transitions: list, [d_t] for all t \in [T]
        d_t = dict, {state_j: (sa_list, probability_list)} for all j \in [S]
            sa_list = [(s_i, a_k)] for all i \in [N_j], k \in [A_j]
                s_i = (z_i, 0)
            probability_list = [P_jikt], 
                P_ijkt is the probability of transitioning to state j by
                taking state-action (s_i, a_k) at time t. 

    """
    pu_action = manhattan.most_neighbors(manhattan.zone_neighbors)
    forward_transitions = []
    backward_transitions = []
    state_list = []
    for z_i in manhattan.zone_neighbors:
        state_list.append((z_i, 0))
    action_dict = {s: [] for s in state_list}
    for s in state_list:
        a_ind = 0
        for neighbor in manhattan.zone_neighbors[s[0]]:
            action_dict[s].append(a_ind)
            a_ind += 1
        # legacy: last action is trying to pick up a rider                
        action_dict[s].append(pu_action) # always add max_action
        
    sa_list = []
    for s in state_list:
        sa_list = sa_list + [(s,a) for a in action_dict[s]]                    
    
    for t_i in transitions_list:
        forward_transitions.append({s:{a:([],[]) for a in action_dict[s]} 
                                    for s in state_list})
        backward_transitions.append({s: ([], []) for s in state_list})
        
        # action for queue_level > 0 is just to drop
        for s in state_list:                                                     
        # action for queue == 0 contains going to neighbors
            # add actions for going to neighbors
            a_ind = -1
            total_neighbors = len(manhattan.zone_neighbors[s[0]])
            for n_j in manhattan.zone_neighbors[s[0]]:  
                # insert actions in forward_transitions
                a_ind += 1
                n_state = (n_j, 0)
                forward_transitions[-1][s][a_ind] = ([n_state], [1])
                # insert actions in backward_transtiions
                backward_transitions[-1][n_state][0].append((s, a_ind))
                backward_transitions[-1][n_state][1].append(1)

                    
                if epsilon > 0 and total_neighbors > 1:
                    forward_transitions[-1][s][a_ind][1][-1] += -epsilon
                    backward_transitions[-1][n_state][1][-1] += -epsilon
                    for n_k in manhattan.zone_neighbors[s[0]]:
                        other_neighbor_state = (n_k, 0)
                        if n_k != n_j:
                            forward_transitions[-1][s][a_ind][0].append(
                               other_neighbor_state)
                            forward_transitions[-1][s][a_ind][1].append(
                                epsilon/(total_neighbors-1))
                            backward_transitions[-1][other_neighbor_state][0].append(
                                    (s, a_ind))
                            backward_transitions[-1][other_neighbor_state][1].append(
                                epsilon / (total_neighbors-1) )
                        
                            # if other_neighbor_state == (79, 0):
                            #     print(f' found {o_ind} -> {other_neighbor_state} in neighbor loop 1')                                              
        for z_i, t_ij in t_i.items():
            o_ind = (z_i, 0) # queue level is 0
            if len(t_ij) == 0: # no rides recorded from z_i at time t_i
            # if no rides are recorded, go back to origin with probability 1.
                forward_transitions[-1][o_ind][pu_action] = ([o_ind], [1])
                backward_transitions[-1][o_ind][0].append((o_ind, pu_action))
                backward_transitions[-1][o_ind][1].append(1)
                
            
            # otherwise for the rides recorded:  
            for dest, p_ij in t_ij.items() :
                # build forward transitions connection
                existing_trans = forward_transitions[-1][o_ind][pu_action]
                flat_dest = (dest[0], 0) 
                # if o_ind == (48,0) and flat_dest == (4,0):
                #     print(f'at {o_ind} -> {flat_dest}')
                #     if flat_dest in existing_trans[0]:
                #         dest_ind = existing_trans[0].index(flat_dest)
                #         cur_forward_val = existing_trans[1][dest_ind]
                #     else:
                #         cur_forward_val = 0
                #     print(f'cur_forward_P = {cur_forward_val}')
                    
                #     existing_backtrans = backward_transitions[-1][flat_dest]
                #     origin_sa = (o_ind, pu_action)
                #     if origin_sa in existing_backtrans:
                #         orig_ind = existing_backtrans[0].index(origin_sa)
                #         cur_back_val = existing_backtrans[1][orig_ind]
                #     else:
                #         cur_back_val = 0
                #     print(f'backward_P = {cur_back_val}')
                    
                if flat_dest in existing_trans[0]:
                    dest_ind = existing_trans[0].index(flat_dest)
                    existing_trans[1][dest_ind] += p_ij
                    
                else:
                    existing_trans[0].append(flat_dest)
                    existing_trans[1].append(p_ij)
                    # if o_ind == (48,0) and flat_dest == (4,0):
                    #     print(f'appended {p_ij}')
                # build backward_transition connection    
                existing_backtrans = backward_transitions[-1][flat_dest]
                origin_sa = (o_ind, pu_action)
                if origin_sa in existing_backtrans[0]:
                    osa_ind = existing_backtrans[0].index(origin_sa)
                    existing_backtrans[1][osa_ind] += p_ij
                else:
                    existing_backtrans[0].append(origin_sa)
                    existing_backtrans[1].append(p_ij)
                    # if o_ind == (48,0) and flat_dest == (4,0):
                    #     print(f' backward append {p_ij}')

           
    for transition in forward_transitions:
        has_queue = [k[1] == 0 for k in transition.keys()]            
        if all(has_queue) == False:
            print(f' found non-zero queue')
        else:
            print('all keys are zero queues')
    # print(f'final backward transition value {backward_transitions[-1][(79,0)][1][3]}  ')
    # print(f'origin state-action is {backward_transitions[-1][(79,0)][0][3]}')            
    return forward_transitions, backward_transitions, state_list, action_dict, sa_list


def transition_kernel_pick_ups(epsilon, transitions_list):
    """

    Parameters
    ----------
    epsilon : float between (0, 1)
        when choosing to go to neighboring state, probability of not 
        getting there.
    transitions_list : list
        Each element of transition_list is t_i.
        t_i: dict: {z_i: t_{ij}}
            z_i =  origin_zone_ind
            t_ij = dict: {dest: p_{ij}}
                dest = (destination zone ind, time)
                p_{ij} = probability of going to dest from z_i. 
    

    Returns
    -------
    forward_transitions: list, [d_i] for all i \in [T]
        d_i = dict, {state_j: t_ij} for all state_j \in [S]
            t_ij = dict, {a_k: P_ijk} for all a_k \in [A_j]
                P_ijk = tuple: (states_list, transitions_list).
                    states_list = list of states (zone_ind, queue_level) to 
                                  transition to
                    transitions_list = list of transition probabilities 
                                       for these states 
                                       at action a_k from state s_j and time i
                                       
                             
    backward_transitions: list, [d_t] for all t \in [T]
        d_t = dict, {state_j: (sa_list, probability_list)} for all j \in [S]
            sa_list = [(s_i, a_k)] for all i \in [N_j], k \in [A_j]
                s_i = (z_i, q_level)
            probability_list = [P_jikt], 
                P_ijkt is the probability of transitioning to state j by
                taking state-action (s_i, a_k) at time t. 

    """
    forward_transitions = []
    backward_transitions = []
    max_action = manhattan.most_neighbors(manhattan.zone_neighbors)
    # legacy: last action is trying to pick up a rider
    for t_i in transitions_list:
        forward_transitions.append({})
        backward_transitions.append({})
        for z_i, t_ij in t_i.items():
            o_ind = (z_i, 0) # queue level is 0
            if o_ind not in forward_transitions[-1]:
                forward_transitions[-1][o_ind] = {}
            if len(t_ij) == 0: # no rides recorded from z_i at time t_i
            # if no rides are recorded, go back to origin with probability 1.
                forward_transitions[-1][o_ind][max_action] = ([o_ind], [1])
                if o_ind not in backward_transitions[-1]:
                    backward_transitions[-1][o_ind] = ([(o_ind, max_action)], [1])
                else:
                    backward_transitions[-1][o_ind][0].append((o_ind, max_action))
                    backward_transitions[-1][o_ind][1].append(1)
            a_ind = -1
            total_neighbors = len(manhattan.zone_neighbors[z_i])
            for n_j in manhattan.zone_neighbors[z_i]:  
                # insert actions in forward_transitions
                a_ind += 1
                n_state = (n_j, 0)
                forward_transitions[-1][o_ind][a_ind] = ([n_state], [1])
                # insert actions in backward_transtiions
                if n_state not in backward_transitions[-1]:
                    backward_transitions[-1][n_state] = ([(o_ind, a_ind)], [1])
                else: 
                    backward_transitions[-1][n_state][0].append((o_ind, a_ind))
                    backward_transitions[-1][n_state][1].append(1)
                    
                if epsilon > 0 and total_neighbors > 1:
                    forward_transitions[-1][o_ind][a_ind][1][-1] += -epsilon
                    backward_transitions[-1][n_state][1][-1] += -epsilon
                    for n_k in manhattan.zone_neighbors[z_i]:
                        other_neighbor_state = (n_k, 0)
                        if n_k != n_j:
                            forward_transitions[-1][o_ind][a_ind][0].append(
                               other_neighbor_state )
                            forward_transitions[-1][o_ind][a_ind][1].append(
                                epsilon/(total_neighbors-1))
                            if other_neighbor_state in backward_transitions[-1]:
                                backward_transitions[-1][other_neighbor_state][0].append(
                                    (o_ind, a_ind))
                                backward_transitions[-1][other_neighbor_state][1].append(
                                    epsilon / (total_neighbors-1) )
                            else:
                                backward_transitions[-1][other_neighbor_state] = (
                                    [(o_ind, a_ind)], [epsilon / (total_neighbors-1)])
                            # if other_neighbor_state == (79, 0):
                            #     print(f' found {o_ind} -> {other_neighbor_state} in neighbor loop 1')
                                
               
            for dest, p_ij in t_ij.items():
                # if dest == (79, 0) and o_ind == (79,1):
                #     print(f' found {o_ind} -> {dest}')
                # build forward transitions
                if max_action not in forward_transitions[-1][o_ind]:
                    forward_transitions[-1][o_ind][max_action] =  ([],[])
                forward_transitions[-1][o_ind][max_action][0].append(dest)
                forward_transitions[-1][o_ind][max_action][1].append(p_ij)
                orig_s = dest
                while orig_s not in forward_transitions[-1] and orig_s[1] > 0:
                    dest_s = (orig_s[0], orig_s[1] - 1)
                    forward_transitions[-1][orig_s] = {
                        max_action: ([dest_s], [1])} # drops queue level with probability 1
                    orig_s = dest_s
                    
                
                # build backward transitions:
                # add the first transition from picking up a driver
                if dest not in backward_transitions[-1]:
                    backward_transitions[-1][dest] = ([],[])
                backward_transitions[-1][dest][0].append((o_ind, max_action))
                backward_transitions[-1][dest][1].append(p_ij)
                # if dest == (79, 0) and o_ind == (79,1):
                #         print(f' found {o_ind} -> {dest} outside while loop')
                        
                    
                # add subsequent transitions from dropping queues
                queue_level = dest[1] - 1
                origin = dest
                dest = (dest[0], queue_level)
                if dest not in backward_transitions[-1]:
                    backward_transitions[-1][dest] = ([],[])
                    
                while queue_level >= 0 and \
                    (origin, max_action) not in backward_transitions[-1][dest][0]:

                    backward_transitions[-1][dest][0].append(
                        (origin, max_action))
                    backward_transitions[-1][dest][1].append(int(1))
                    # if dest == (79, 0) and origin == (79,1):
                    #     print(f' found {origin} -> {dest} at {transitions_list.index(t_i)}')
                    #     print(f' value added is {backward_transitions[-1][dest][1][-1]}')
                    origin = dest   
                    queue_level += -1
                    dest = (origin[0], queue_level)
                    if dest not in backward_transitions[-1] and queue_level >= 0:
                        backward_transitions[-1][dest] = ([],[])
                    
                    
    # print(f'final backward transition value {backward_transitions[-1][(79,0)][1][3]}  ')
    # print(f'origin state-action is {backward_transitions[-1][(79,0)][0][3]}')            
    return forward_transitions, backward_transitions


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