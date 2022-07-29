# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:07:40 2022

@author: Sarah Li
"""
import models.taxi_dynamics.manhattan_transition as m_trans
import models.taxi_dynamics.manhattan_cost as m_cost
import pickle
import pandas as pd
import numpy as np


directory = 'C:/Users/craba/Desktop/code/mdpcg/V3/' 
trips_filename = directory+'models/manhattan_transitions.pickle'


class queue_game:
    
    def __init__(self, total_mass = 1, epsilon=0.01, strictly_convex = True):
        self.mass = total_mass
        trips_file = open(trips_filename, 'rb')
        m_transitions = pickle.load(trips_file)
        for transition in m_transitions:
            transition.pop(103)
            transition.pop(104)
            transition.pop(105)
        trips_file.close()
        mdp = m_trans.transition_kernel_dict(epsilon, m_transitions)
        self.forward_P = mdp[0]
        self.backward_P = mdp[1]
        self.state_list = mdp[2]
        self.action_dict = mdp[3]
        self.sa_list = mdp[4]
        
        # self.forward_P, self.backward_P = m_trans.transition_kernel_pick_ups(
        #     epsilon, m_transitions)
        T = 12
        S = 63
        # there's a states mismatch somewhere 
        # - need to regenerate the count and weighted average files
        demand_rate = m_cost.demand_rate(
            directory+'models/taxi_dynamics/count_kernel_jan.csv', T, S)
        self.avg_dist = pd.read_csv(
            directory + 'models/taxi_dynamics/weighted_average_jan.csv', 
            header=None).values
        
        self.costs = m_cost.congestion_cost_dict(
            demand_rate, self.forward_P, self.avg_dist, epsilon=1e-3)
        self.transition_data = m_transitions

    def get_potential(self, density):
        return sum([sum([0.5*self.costs[t][st][0] * density[t][st]**2 \
                      + self.costs[t][st][0] * density[t][st] 
                      for st in self.costs[t].keys()]) 
                    for t in range(len(self.costs))]) 
        
    def get_gradient(self, density):
        grad = []
        for t in range(len(self.costs)):
            grad.append({})
            for st in self.costs[t].keys():
                grad[st] = self.costs[t][st][0]*density[t][st] + \
                    self.costs[t][st][1] 
        return grad

    def random_initial_density(self):
        initial_density = []
        # at time 0, randomly generate a density that satisfies
        # transition dynamics
        cur_density = {s: np.random.random() for s in self.state_list}
        density_sum = sum(cur_density.values())
        scaling = self.mass / density_sum
        cur_density = {s: d*scaling for s, d in cur_density.items()}
        for t in range(len(self.forward_P)):
            initial_density.append({})
            for s in cur_density.keys():
                # print(f' current state {s} t = {t}')
                policy = [np.random.random() for _ in self.action_dict[s]]
                policy_scale = sum(policy)
                policy = [p / policy_scale for p in policy]
                assert round(sum(policy), 5) == 1, f'policy sums to {round(sum(policy), 5)}'
                for a_ind in range(len(self.action_dict[s])):
                    # print(f'is {s} in density list '
                    #       f'{s in cur_density} at t = {t}')
                    initial_density[-1][(s,  self.action_dict[s][a_ind])] = \
                        policy[a_ind] * cur_density[s]
                sa_sum = sum([initial_density[-1][(s, a)] for a in self.action_dict[s]])
                assert round(sa_sum - cur_density[s], 5) == 0, \
                    f' at state {s}, sum {sa_sum} != {cur_density[s]}'
            # assert round(sum(initial_density[-1].values()), 5) == 1, \
            #     f' in random_initial: density at time 0 is {sum(initial_density[-1].values())}'

            cur_density = self.propagate(initial_density[-1], t)
            # if t==4:
            #     print(f'current density {cur_density[(127, 1)]}')
        return initial_density
       
    def get_probability(self, sa_density, t):
        p_density = {}
        transition_t = self.forward_P[t]
        for s in transition_t.keys():
            p_density[s] = sum([sa_density[(s,a)] 
                                for a in transition_t[s].keys()])

        return p_density
            
    def propagate(self, sa_density, t):
        transition_t = self.backward_P[t]
        next_density = {}
        for s in transition_t.keys():
            
            orig_s, probs = transition_t[s]
            next_density[s] = sum([probs[i] * sa_density[orig_s[i]] 
                                   for i in range(len(probs))])
            # if s == (127, 1) and t == 4:
            #     print(f'{s} in backward_P at t = {t}')
            #     print(f'came from {orig_s}')
            #     print(f'came with probabilities {probs}')
            #     for i in range(len(probs)):
            #         print(f'densities at {orig_s[i]} is {sa_density[orig_s[i]]}')
        # print(f' at time {t}, is (88, 5) in next_density {(88,5) in next_density}')
        return next_density
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                