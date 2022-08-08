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
month = 'jan' # 'dec' # 
ints = 12 # 15 min
trips_filename = directory+f'models/taxi_data/manhattan_transitions_{month}_{ints}min.pickle'
count_filename = directory+f'models/taxi_data/count_kernel_{month}_{ints}min.csv'
avg_filename = directory +f'models/taxi_data/weighted_average_{month}_{ints}min.csv'
class queue_game:
    
    def __init__(self, total_mass = 1, epsilon=0.1, 
                 strictly_convex=True, uniform_density=False, flat=False):
        self.mass = total_mass
        trips_file = open(trips_filename, 'rb')
        m_transitions = pickle.load(trips_file)
        trips_file.close()
        if flat:
            mdp = m_trans.transition_kernel_dict_flat(epsilon, m_transitions)
        else:
            mdp = m_trans.transition_kernel_dict(epsilon, m_transitions)
        self.forward_P = mdp[0]
        self.backward_P = mdp[1]
        self.state_list = mdp[2]
        self.action_dict = mdp[3]
        self.sa_list = mdp[4]
        self.z_list = [s[0] for s in self.state_list]
        self.z_list = list(set(self.z_list)) # get unique values from z_list
        print(f' number of zones {len(self.z_list)}')
        print(f' number of states {len(self.state_list)}')
        self.T = len(self.forward_P)
        S = len(self.z_list)
        print(f' length of time horizon is {self.T}')
        self.max_q = 8 if self.T == 15 else 7
        self.constrain_queue = None
        
        self.t0_density = self.uniform_t0_density() if uniform_density \
            else self.random_t0_density()
        self.tolls = None
        
        
        # there's a states mismatch somewhere 
        # - need to regenerate the count and weighted average files
        demand_rate = m_cost.demand_rate(count_filename, self.T, S)
        self.avg_dist = pd.read_csv(avg_filename, header=None).values
        
        self.costs = m_cost.congestion_cost_dict(
            demand_rate, self.forward_P, self.avg_dist, epsilon=1e-3)
        self.transition_data = m_transitions
        self.constrained_states = None
        self.constrained_val = None
        
    def get_strong_convexity(self):
        min_R = 100
        for t in range(len(self.costs)):
            min_R = min([r[0] if r[0] != 0 else 999999 
                         for r in self.costs[t].values()] + [min_R])
        return min_R
    
    def get_social_cost(self, density):
        potential_val = sum([sum([self.costs[t][st][0] * density[t][st]**2 \
                + self.costs[t][st][0] * density[t][st] 
                for st in self.costs[t].keys()]) 
                for t in range(len(self.costs))]) 
        if self.tolls is not None:
            for zt in self.tolls.keys():
                z_ind = zt[0]
                t_ind = zt[1]
                for a in self.action_dict[z_ind]:
                    potential_val += self.tolls[zt]*density[t_ind][(z_ind, a)]
            
        return potential_val
    def get_potential(self, density):
        potential_val = sum([sum([  
            0.5*self.costs[t][st][0] * density[t][st]**2 \
                + self.costs[t][st][1] * density[t][st] 
                for st in self.costs[t].keys()]) 
                for t in range(len(self.costs))]) 
        if self.tolls is not None:
            for zt in self.tolls.keys():
                z_ind = zt[0]
                t_ind = zt[1]
                for a in [7]:# self.action_dict[z_ind]:
                    potential_val += self.tolls[zt]*density[t_ind][(z_ind, a)]
            
        return potential_val
        
    def get_gradient(self, density):
        grad = []
        for t in range(len(self.costs)):
            grad.append({})
            for st in self.costs[t].keys():
                grad[-1][st] = self.costs[t][st][0]*density[t][st] + \
                    self.costs[t][st][1] 
        if self.tolls is not None:
            for zt in self.tolls.keys():
                for a in [7]: # self.action_dict[zt[0]]:
                    grad[zt[1]][(zt[0], a)] += self.tolls[zt]
                
        return grad
    
    def uniform_t0_density(self):
        t0_density = {(z,0): self.mass/len(self.z_list) for z in self.z_list}
        for q in range(1, self.max_q):
            t0_density.update({(z,q): 0 for z in self.z_list})
        return t0_density
        
    def random_t0_density(self):
        t0_density = {s: np.random.random() 
                      for s in self.state_list}
        density_sum = sum(t0_density.values())
        scaling = self.mass / density_sum
        t0_density = {s: d*scaling for s, d in t0_density.items()}
        return t0_density
    
    
    def whole_length_density(self):
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
        return next_density

    def get_zone_densities(self, sa_density, include_queues=False):
        """ Get the drivers not in queue in each zone 
            (collapse queue levels and actions).
        
        Input:
            sa_density: list. [d_i] for i = 0...T-1
                d_i: dict. {sa: d_isa} for sa in State x Actions.
                    d_isa: float. Density in state-action sa at time i
                    States: (zone_ind, queue_level)
            include_queues: bool. True if returned density should include all
                                 queues as well. 
        Output:
            z_density: list. [z_i] for i = 0...T-1
                z_i: dict. {z_ind: d_iz} for z_ind in Zone inds.
                    d_iz: float. Density in zone  z_ind at time t.
        """
        z_density = [{z: 0 for z in self.z_list} for _ in sa_density]
        for i in range(len(sa_density)):
            for sa, d_isa in sa_density[i].items():
                if include_queues:
                    z_density[i][sa[0][0]] += d_isa
                elif sa[0][1] == 0:   
                    z_density[i][sa[0][0]] += d_isa
                
        return z_density                
                
    def update_tolls(self, tau):
        self.tolls = {}
        for k in tau.keys():
            self.tolls[k] = tau[k]
            
    def set_constraints(self, constrained_zones, constrained_val, 
                        with_queue=False):
        self.constrained_val = constrained_val
        self.constrain_queue = with_queue
        self.constrained_states = []
        for c in constrained_zones:
            max_q =  self.max_q if with_queue else 1
            for q in range(max_q):
                self.constrained_states = self.constrained_states + \
                    [(c, q) for q in range(max_q)]
                    
    def get_constrained_gradient(self, density, return_violation=False):
        gradient = {}
        for t in range(self.T):
            for s in self.constrained_states:
                # if self.constraine_queue:
                total_density = sum([density[t][(s, a)] 
                                     for a in self.action_dict[s[0]]])
                gradient[(s,t)] = total_density - self.constrained_val
        if return_violation:
            grad_arr = np.array(list(gradient.values()))
            grad_arr[grad_arr < 0] = 0
            return gradient, np.linalg.norm(grad_arr, 2)
        else:
            return gradient
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                