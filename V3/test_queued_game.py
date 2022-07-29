# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:47:27 2022

@author: Sarah Li
"""
import models.taxi_dynamics.manhattan_neighbors as m_neighs
import numpy as np
import models.queued_mdp_game as game



            
mass = 100
manhattan_game = game.queue_game(mass, 0.1)


#%% Test Transition dynamics %%%
# test queued game: 
forward_P = manhattan_game.forward_P
backward_P = manhattan_game.backward_P

# test forward transition dynamics
assert type(forward_P) is list, 'forward transition should be a list'
assert type(backward_P[0]) is dict, 'each element on list is a dictionary'

checked_state = {}
check_ind = np.random.randint(0, len(forward_P))
pu_action = m_neighs.most_neighbors(m_neighs.STATE_NEIGHBORS)
for tup in forward_P[check_ind].keys():
    
    if  tup[0] in checked_state: # (X, N) has been checked
        if checked_state[tup[0]]  > tup[1]:
            continue
    # if (X, N) is a state, (X, N-1) is a state
    queue_level = tup[1]
    bottom_level = 0 if tup[0] not in checked_state else checked_state[tup[0]]
    while queue_level >= bottom_level:
        lower_queue =  (tup[0], queue_level)
        assert lower_queue in forward_P[check_ind].keys(), \
            f'queue level {lower_queue} is missing'
        queue_level += -1
    checked_state[tup[0]] = tup[1]
        
    # action to pick up riders  is in each state
    assert pu_action in forward_P[check_ind][tup].keys(), \
        f'{pu_action} not in {forward_P[check_ind][tup].keys()} at tuple {tup}, ' \
        f'at t = {check_ind}'
        
    # assert all other actions correspond to going to some neighbor
    for a_ind in forward_P[check_ind][tup].keys():
        if a_ind != pu_action:
            neighbors, probs = forward_P[check_ind][tup][a_ind] 
            # print(f' tup is {tup} action is {a_ind}')
            cur_neighbor = m_neighs.zone_neighbors[tup[0]][a_ind]
            assert cur_neighbor == neighbors[0][0], \
                f' at state {tup}, neighbor {cur_neighbor} is not {neighbors[0][0]} '
        # sum of probabilities is 1 for each state action:
        dests =  forward_P[check_ind][tup][a_ind]
        assert round(sum(dests[1]), 5) == 1, \
            f'transitions for {tup} sums to {sum(dests[1])} != 1'
        # check that all destination states are states and neighbors
        for dest_state in dests[0]:
            assert dest_state in forward_P[check_ind].keys(), \
            f'{dest_state} not in state list'
            if a_ind != pu_action:
                assert dest_state[0] in m_neighs.zone_neighbors[tup[0]], \
                f'at {tup[0]}, destination {dest_state[0]} not in ' \
                f'neighbor list {m_neighs.zone_neighbors[tup[0]]}'
# test for all other actions:
    
print('all tests for forward transitions passed')

# test backward_transition_dynamics
assert type(backward_P) is list, 'backward transition should be a list'
assert type(backward_P[0]) is dict, 'each element of backward_transition is a dict'

check_ind = np.random.randint(0, len(forward_P))
for dest, orig in backward_P[check_ind].items():
    # check the original pair exists in forward_P:
    for i in range(len(orig[0])):
        o_state = orig[0][i][0]
        o_action = orig[0][i][1]
        dest_ind = forward_P[check_ind][o_state][o_action][0].index(dest)
        forward_probability = forward_P[check_ind][o_state][o_action][1][dest_ind]
        # check that probabilities in the forward and backward cases match
        assert orig[1][i] == forward_probability, \
        f'dest {dest} to origin {o_state} probabilities mismatch ' \
        f'{orig[1][i]} != {forward_probability} at time {check_ind}'
print('all tests for backward transition passed')
    

# testing random trips are stored for each file. 



#%% Test the initial densities %%#

# np.random.seed(111)
initial_density = manhattan_game.random_initial_density()
# check that at each time, density sums to 1
for t in range(len(initial_density)):
    density_t = initial_density[t]
    density_sum = sum([sum([density_t[(s,a)] 
                            for a in manhattan_game.action_dict[s]]) 
                for s in manhattan_game.state_list])
    assert round(density_sum, 5) == mass, f' density at time {t} is {density_sum}'


# check that forward propagation works too: 
total_sample = 1000
for sample in range(total_sample):
    cur_state_ind = np.random.choice([i for i in range(len(manhattan_game.state_list))])
    cur_state = manhattan_game.state_list[cur_state_ind]
    cur_t = np.random.randint(1, 11)
    cur_transition = forward_P[cur_t-1]
    last_density = initial_density[cur_t-1]
    cur_density = sum([initial_density[cur_t][(cur_state,a)] 
                       for a in manhattan_game.action_dict[cur_state]])
    test_density = 0
    for s in manhattan_game.state_list:
        for a in manhattan_game.action_dict[s]:
            if cur_state in cur_transition[s][a][0]:
                cur_ind = cur_transition[s][a][0].index(cur_state)
                test_density += last_density[(s,a)] * cur_transition[s][a][1][cur_ind]
                # if cur_state == (127, 1) and cur_t == 4:
                #     print(f'{s} goes to {cur_state} with p = {round(cur_transition[s][a][1][cur_ind],2)}'
                #           f'and density is  {last_density[(s,a)]}')

    assert round(test_density - cur_density, 3) == 0, \
    f'density at s = {cur_state}, t= {cur_t} does not match: ' \
    f'{test_density} != {cur_density}'
print('all tests for density transition passed')
 


#%% Test cost functions %%#
obj = manhattan_game.get_potential(initial_density)





















