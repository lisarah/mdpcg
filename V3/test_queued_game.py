# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:47:27 2022

@author: Sarah Li
"""
import models.taxi_dynamics.manhattan_neighbors as m_neighs
import numpy as np
import models.queued_mdp_game as game
import algorithm.dynamic_programming as dp
import algorithm.FW as fw
import matplotlib.pyplot as plt
import time
import models.test_model as test

            
mass = 10000
flat = False
is_test = False
if is_test: 
    neighbor_list = {1: [1, 2], 2: [2, 1]}
    manhattan_game = test.queue_game(mass, uniform_density=True)
else:
    neighbor_list = m_neighs.zone_neighbors
    manhattan_game = game.queue_game(mass, 0.1, uniform_density=True, 
                                     flat=flat)


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
            cur_neighbor = neighbor_list[tup[0]][a_ind]
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
                assert dest_state[0] in neighbor_list[tup[0]], \
                f'at {tup[0]}, destination {dest_state[0]} not in ' \
                f'neighbor list {neighbor_list[tup[0]]}'
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
        if flat:
             assert orig[1][i] >=forward_probability, \
            f'dest {dest} to origin {o_state} probabilities mismatch ' \
            f'{orig[1][i]} is less than {forward_probability} at time {check_ind}'
        else:
            
            assert orig[1][i] == forward_probability, \
            f'dest {dest} to origin {o_state} probabilities mismatch ' \
            f'{orig[1][i]} != {forward_probability} at time {check_ind}'
print('all tests for backward transition passed')

#%% Test the initial densities %%#
def test_density(d_list, mass, game):
    
    # check that at each time, density sums to 1
    for t in range(len(d_list)):
        d_sum = sum([sum([d_list[t][(s,a)] for a in game.action_dict[s]]) 
                     for s in game.state_list])
        assert round(d_sum, 5) == mass, f' density at time {t} is {d_sum}'
    # check that forward propagation works too: 
    total_sample = 100
    for sample in range(total_sample):
        s_ind = np.random.choice([i for i in range(len(game.state_list))])
        cur_s = manhattan_game.state_list[s_ind]
        cur_t = np.random.randint(1, 11)
        cur_trans = forward_P[cur_t-1]
        last_density = d_list[cur_t-1]
        cur_density = sum([d_list[cur_t][(cur_s,a)] 
                           for a in game.action_dict[cur_s]])
        test_d = 0
        for s in game.state_list:
            for a in game.action_dict[s]:
                if cur_s in cur_trans[s][a][0]:
                    cur_ind = cur_trans[s][a][0].index(cur_s)
                    test_d += last_density[(s,a)] * cur_trans[s][a][1][cur_ind]
    
        assert round(test_d - cur_density, 3) == 0, \
        f'density at s = {cur_s}, t= {cur_t} does not match: ' \
        f'{test_d} != {cur_density}'
    print('all tests for density passed')
    
    
# np.random.seed(111)
print('testing for current density')
initial_density = manhattan_game.get_density()
test_density(initial_density, mass, manhattan_game)
 


#%% Test cost functions %%#
obj = manhattan_game.get_potential(initial_density)
grad = manhattan_game.get_gradient(initial_density)
V, pol = dp.value_iteration_dict(grad, forward_P)

next_sa, next_s = dp.density_retrieval(pol, manhattan_game)

print(f'random_objective = {obj}')
print(f'new_objective    = {manhattan_game.get_potential(next_sa)}')

print('testing for next state-action density')
test_density(next_sa, mass, manhattan_game)
print('testing for next state density')
for t in range(len(next_s)):
    d_sum = sum([next_s[t][s] for s in manhattan_game.state_list])
    assert round(d_sum, 5) == mass, f' density at time {t} is {d_sum}'
    
# test for the maximum multiplier and minimum multiplier of the game
max_R = 1
min_R = 100
for t in range(len(manhattan_game.costs)):
    max_R = max([r[0] for r in manhattan_game.costs[t].values()] + [max_R])
    min_R = min([r[0] if r[0] != 0 else 999999 
                 for r in manhattan_game.costs[t].values()] + [min_R])

print(f' max r is {max_R}')
print(f' min r is {min_R}')
#%% test Frank Wolfe %%#
begin_t = time.time()
y_res, obj_hist = fw.FW_dict(manhattan_game, 
                             max_error=100, max_iterations=1e3)
end_t = time.time()
print(f' total FW time = {end_t - begin_t}')
plt.figure()
plt.title('Potential value as function of algorithm iteration')
plt.plot(obj_hist)
plt.grid()
plt.show()

    
#%% test that final density is still equal to mass %%#
congested_zones = [161, 261, 87]
# congested_zones = [1, 2]
congested_densities = []
z_densities = manhattan_game.get_zone_densities(y_res[-1], include_queues=True)
for t in range(len(z_densities)):
    assert round(sum(z_densities[t].values()), 5) == mass, \
    f'at time {t}, total densities is {round(sum(z_densities[t].values()), 5) }'
    print(f' zone density at time {t} = {round(sum(z_densities[t].values()), 5)}')
    c_t = {c: z_densities[t][c] for c in congested_zones}
    congested_densities.append(c_t)

plt.figure()
for c in congested_zones:
    congestion_in_t = [c_t[c] for c_t in congested_densities]
    plt.plot(congestion_in_t, label = c)
plt.grid()
plt.legend()
plt.show()












