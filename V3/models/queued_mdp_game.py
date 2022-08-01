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
trips_filename = directory+'models/manhattan_transitions_jan.pickle'


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
        self.z_list = [s[0] for s in self.state_list]
        self.z_list = list(set(self.z_list)) # get unique values from z_list
        self.t0_density = self.random_t0_density()
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
                grad[-1][st] = self.costs[t][st][0]*density[t][st] + \
                    self.costs[t][st][1] 
        return grad
    
    def random_t0_density(self):
        t0_density = {s: np.random.random() 
                      for s in self.state_list}
        density_sum = sum(t0_density.values())
        scaling = self.mass / density_sum
        t0_density = {s: d*scaling for s, d in t0_density.items()}
        return t0_density
    
    def random_density(self):
        initial_s_density = [self.t0_density]
        initial_sa_density = []
        # at time 0, randomly generate a density that satisfies
        # transition dynamics
        for t in range(len(self.forward_P)):
            s_density = initial_s_density[-1]
            initial_sa_density.append({})
            sa_density = initial_sa_density[-1]
            for s in s_density.keys():
                # print(f' current state {s} t = {t}')
                policy = [np.random.random() 
                          for _ in self.action_dict[s]]
                policy_scale = sum(policy)
                policy = [p / policy_scale for p in policy]
                for a_ind in range(len(self.action_dict[s])):
                    cur_act = self.action_dict[s][a_ind]
                    sa_density[(s,cur_act) ] = \
                        policy[a_ind] * s_density[s]
            initial_s_density.append(
                self.propagate(sa_density, t))
        return initial_sa_density
       
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

    def get_zone_densities(self, sa_density):
        """ Get the total density in each zone 
            (collapse queue levels and actions).
        
        Input:
            sa_density: list. [d_i] for i = 0...T-1
                d_i: dict. {sa: d_isa} for sa in State x Actions.
                    d_isa: float. Density in state-action sa at time i
                    States: (zone_ind, queue_level)
        Output:
            z_density: list. [z_i] for i = 0...T-1
                z_i: dict. {z_ind: d_iz} for z_ind in Zone inds.
                    d_iz: float. Density in zone  z_ind at time t.
        """
        z_density = [{z: 0 for z in self.z_list} for _ in sa_density]
        for i in range(len(sa_density)):
            for sa, d_isa in sa_density[i].items():
                z_density[i][sa[0][0]] += d_isa
        return z_density                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                