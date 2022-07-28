# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:47:27 2022

@author: Sarah Li
"""
import models.taxi_dynamics.manhattan_transition as m_trans
import models.taxi_dynamics.manhattan_neighbors as m_neighs
import pickle
import numpy as np


directory = 'C:/Users/craba/Desktop/code/mdpcg/V3/' 
trips_filename = directory+'models/manhattan_transitions.pickle'



class quad_game:
    
    def __init__(self, strictly_convex = True):
        trips_file = open(trips_filename, 'rb')
        m_transitions = pickle.load(trips_file)
        for transition in m_transitions:
            transition.pop(103)
            transition.pop(104)
            transition.pop(105)
        trips_file.close()
        self.forward_P, self.backward_P = m_trans.transition_kernel_pick_ups(
            0.01, m_transitions)
        self.transition_data = m_transitions
    

manhattan_game = quad_game()

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
    
    # sum of probabilities is 1 for each state action:
    dests =  forward_P[check_ind][tup][pu_action]
    for dest_state in dests[0]:
        assert dest_state in forward_P[check_ind].keys(), f'{dest_state} not in state list'
    assert round(sum(dests[1]), 5) == 1, f'transitions for {tup} to {sum(dests[1])}'
    
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
    

