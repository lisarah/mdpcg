# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 21:22:51 2022

@author: Sarah Li
"""
import numpy as np


class queue_game:
    def __init__(self, total_mass=1, eps=0.01, uniform_density=True):
        forward_P_t = {
            (1, 0): {0:([(1,0), (2,0)], [0.2, 0.8]),
                     7:([(1,0), (2,0)], [0.9, 0.1])
                     },
            (2, 0): {0:([(2,0)], [1.0]),
                     7:([(1,0)], [1.0])
                     } # 
            }
        self.mass = total_mass
        self.T = 15
        self.forward_P = [forward_P_t for _ in range(self.T)]
        backward_P_t = {
            (1, 0):([((1,0), 0), ((1,0), 7), ((2,0), 7)], [0.2, 0.9, 1.]),
            (2, 0):([((1,0), 0), ((1,0), 7), ((2,0), 0)], [0.8, 0.1, 1.])
            }
        self.backward_P = [backward_P_t for _ in range(self.T)]
        
        self.state_list = [(1,0), (2,0)]
        self.sa_list = [((1,0), 0), ((2,0), 0), ((1,0), 7), ((2,0), 7)]
        costs_t = {((1,0), 0): [1., -2.],
                   ((1,0), 7): [0.5, -4], 
                   ((2,0), 0): [2., -3],
                   ((2,0), 7): [1, -3]}
        self.action_dict = {(s, 0): [0, 7] for s in [1,2]}
        self.costs = [costs_t for _ in range(self.T)]
        self.max_q = 1
        self.t0 = self.t0_density(uniform_density) 
        self.tolls = None
        
    def get_potential(self, density):
        potential_val = sum([sum([0.5*c_t[sa][0]*d_t[sa]**2+c_t[sa][1]*d_t[sa]
                                  for sa in self.sa_list])
                             for c_t, d_t in zip(self.costs, density)])
        if self.tolls is not None:
            potential_val += sum([
               toll*(sum([density[zt[1]][(zt[0], a)] 
                          for a in self.action_dict[zt[0]]]) \
                     - self.constrained_val)
                for zt, toll in self.tolls.items()])
        return potential_val
    
    def get_social_cost(self, density):
        potential_val = sum([sum([c_t[sa][0]*d_t[sa]**2+c_t[sa][1]*d_t[sa]
                                  for sa in self.sa_list])
                             for c_t, d_t in zip(self.costs, density)])
        if self.tolls is not None:
            potential_val += sum([
               toll*(sum([density[zt[1]][(zt[0], a)] 
                          for a in self.action_dict[zt[0]]]) \
                     - self.constrained_val)
                for zt, toll in self.tolls.items()])        
        return potential_val 
    
    def get_strong_convexity(self):
        return 0.5
    
    def get_gradient(self, density):
        gradient = [{sa: c_t[sa][0]*d_t[sa]+c_t[sa][1] for sa in self.sa_list} 
                    for c_t, d_t in zip(self.costs, density)]
        if self.tolls is not None:
            for st in self.tolls.keys():
                t = st[1]
                s = st[0]
                for a in self.action_dict[s]:
                    gradient[t][(s,a)] += self.tolls[st]
        return gradient
    
    def t0_density(self, uniform_density):
        if uniform_density:
            t0 = {s: self.mass/len(self.state_list) for s in self.state_list}
        else:
            t0 = {s: np.random.random() for s in self.state_list}
            scaling = self.mass / sum(t0.values())
            t0 = {s: d*scaling for s, d in t0.items()}
        # initialize all mass out of queues
        for q in range(1, self.max_q):
            t0.update({(s,q): 0 for s in self.state_list})

        return t0
    
    def get_density(self, random_policy=True):
        initial_s_density = [self.t0]
        initial_sa_density = [{} for _ in range(self.T)]
        # at time 0, randomly generate a density that satisfies
        # transition dynamics
        for t in range(self.T):
            s_density = initial_s_density[-1]
            d_t = initial_sa_density[t]
            for s in s_density.keys():
                # print(f' current state {s} t = {t}')
                policy = [np.random.random() for _ in self.action_dict[s]]
                p_scale = sum(policy)
                policy = [p/p_scale for p in policy]
                d_t.update({(s, a): pol_a* s_density[s] 
                            for pol_a, a in zip(policy, self.action_dict[s])})
                
            initial_s_density.append(self.propagate(d_t, t))
        return initial_sa_density
    
    def propagate(self, sa_density, t):
        # for s, val in self.backward_P[t].items():
        #     for orig_s, prob in zip(val[0],val[1]):
        #         print(f' orig_s {orig_s}')
        #         print(f' prob {prob}')
        # print(f'sa density {sa_density}')
        next_density = {s: sum([prob* sa_density[orig_s] 
                                for orig_s, prob in zip(val[0],val[1])]) 
                        for s, val in self.backward_P[t].items()}
        return next_density
    
    
    def get_zone_densities(self, sa_density,include_queues=False):
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
        flat_density = [{s[0]: sum([d_t[(s,a)] for a in self.action_dict[s]]) 
                         for s in self.state_list} for d_t in sa_density]
      
                
        return flat_density     
    
    def set_constraints(self, constrained_zones, constrained_val, 
                        with_queue=False):
        self.constrained_val = constrained_val
        self.constrain_queue = with_queue
        self.constrained_states = [(z, 0) for z in constrained_zones]
        
                    
    def get_constrained_gradient(self, density, return_violation=False):
        gradient = {}
        for t in range(self.T):
            for s in self.constrained_states:
                # if self.constraine_queue:
                total_density = sum([density[t][(s, a)] 
                                     for a in self.action_dict[s]])
                gradient[(s,t)] = total_density - self.constrained_val
        if return_violation:
            grad_arr = np.array(list(gradient.values()), copy=True)
            grad_arr[grad_arr < 0] = 0
            return gradient, np.linalg.norm(grad_arr, 2)
        else:
            return gradient
        
    def update_tolls(self, tau):
        self.tolls = {}
        for k in tau.keys():
            self.tolls[k] = tau[k]   