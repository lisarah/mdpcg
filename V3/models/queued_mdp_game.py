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


directory = ''  
# 'C:/Users/craba/Desktop/code/mdpcg/V3/'  
# 'C:/Users/Sarah Li/Desktop/code/mdpcg/V3/' 
month = 'jan' # 'dec' # 
ints = 15 # 15 min
trips_filename = directory+f'models/taxi_data/manhattan_transitions_{month}_{ints}min.pickle'
count_filename = directory+f'models/taxi_data/count_kernel_{month}_{ints}min.csv'
avg_filename = directory +f'models/taxi_data/weighted_average_{month}_{ints}min.csv'
class queue_game:
    
    def __init__(self, total_mass = 1, epsilon=0.1, 
                 strictly_convex=True, uniform_density=False, flat=False):
        """  Initialize a queued MDP game for rideshare drivers. The  
        transition dynamics and costs are built on the ride demand data from
        New York City's Taxi and Limousine Commission.
        https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
        
        
        Parameters
        ----------
        total_mass : float, optional
            Mass of the driver fleet. The default is 1.
        epsilon : float, optional
            DESCRIPTION. The default is 0.1.
        strictly_convex : TYPE, optional
            DESCRIPTION. The default is True.
        uniform_density : TYPE, optional
            DESCRIPTION. The default is False.
        flat : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.mass = total_mass
        trips_file = open(trips_filename, 'rb')
        m_transitions = pickle.load(trips_file)
        trips_file.close()
        self.flat = flat
        if self.flat:
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
        self.T = len(self.forward_P)
        
        print(f' number of zones {len(self.z_list)}')
        print(f' number of states {len(self.state_list)}')
        print(f' length of time horizon is {self.T}')
        self.max_q = 8 if self.T == 15 else 7
        if self.flat:
            self.max_q = 1
        self.constrain_queue = None
        
        self.t0 = self.t0_density(uniform_density) 
        self.tolls = None

        demand_rate = m_cost.demand_rate(count_filename, self.T,  len(self.z_list))
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
    
    def t0_density(self, uniform_density):
        if uniform_density:
            # uniformly initialize density in the zeroth states
            t0 = {(z, 0): self.mass/len(self.z_list) for z in self.z_list}
        else:
            t0 = {(z, 0): np.random.random() for z in self.z_list}
            scaling = self.mass / sum(t0.values())
            t0 = {s: d*scaling for s, d in t0.items()}
        # initialize all mass out of queues
        if not self.flat:
            for q in range(1, self.max_q):
                t0.update({(z,q): 0 for z in self.z_list})

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
                if random_policy:
                    policy = [np.random.random() for _ in self.action_dict[s]]
                    p_scale = sum(policy)
                    policy = [p/p_scale for p in policy]
                else:
                    assert False, 'Non random policy not implemented'
                d_t.update({(s, a): pol_a* s_density[s] 
                            for pol_a, a in zip(policy, self.action_dict[s])})
                
            initial_s_density.append(self.propagate(d_t, t))
        return initial_sa_density

            
    def propagate(self, sa_density, t):
        
        next_density = {s: sum([prob* sa_density[orig_s] 
                                for orig_s, prob in zip(val[0],val[1])]) 
                        for s, val in self.backward_P[t].items()}
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
        if not include_queues:
            z_density = [{s: sum([d_t[(s,a)] for a in self.action_dict[s]]) 
                          for s in self.state_list} for d_t in sa_density]
        else:
            z_density = [{z: sum([
                sum([d_t[((z,q),a)] for a in self.action_dict[(z,q)]])
                for q in range(self.max_q)]) for z in self.z_list} 
                for d_t in sa_density]
        return z_density   
             
    def get_average_density(self, z_density):
        if self.flat:
            avg_density = {
                z: sum([z_density[t][z] for t in range(self.T)])/self.T 
                for z in self.z_list}
        else:
            avg_density = {
                z: sum([z_density[t][(z, 0)] for t in range(self.T)])/self.T 
                for z in self.z_list}
        return avg_density
    
    def get_violations(self, z_density, c_val, return_density=True):
        constraint_violation= {}
        v_density = {}
        for z in z_density[0].keys():
            threshold = [max(0, z_density[t][z]-c_val) for t in range(self.T)]
            violation =  np.linalg.norm(np.array(threshold), 2) #sum([v for v in threshold if v > 0])
            if violation > 0:
                if self.flat:
                    constraint_violation[z] = violation
                else:
                    # this should only happen once for the 0 level queue
                    constraint_violation[z[0]] = violation
                print(f'zone {z} violates constraint <{c_val} by {violation}')
                if return_density:
                    v_density[z] = [z_density[t][z] for t in range(self.T)]
        if return_density:
            return constraint_violation, v_density
        else:
            return constraint_violation
    
    def get_violation_subset(self, z_density, states, c_val):
        if type(states[0]) == tuple and self.flat:
        # z_density is flat but states are given in tuples
            v_key = [s[0] for s in states] 
            z_key = [s[0] for s in states] 
        elif type(states[0]) != tuple and not self.flat:
        # z_density has tuple keys but states is given in zones
            v_key = states 
            z_key = [(s,0) for s in states]
            
        v_density = {v: [z_density[t][z] for t in range(self.T)] 
                     for v, z in zip(v_key, z_key)}
        constraint_violation = {
            v: sum([max(d_zt-c_val, 0) for d_zt in v_density[v]])/self.T
            for v in v_density.keys()}

        return v_density, constraint_violation
        
    def update_tolls(self, tau):
        self.tolls = {k: tau_k for k, tau_k in tau.items()}
            
    def set_constraints(self, constrained_zones, constrained_val, 
                        with_queue=False):
        self.constrained_val = constrained_val
        self.constrain_queue = with_queue
        self.constrained_states = []
        self.constrained_states = [(z, 0) for z in constrained_zones]
                    
    def get_constrained_gradient(self, density, return_violation=False):
        gradient = {}
        for t in range(self.T):
            d_t = density[t]
            for s in self.constrained_states:
                total_density = sum([d_t[(s,a)] for a in self.action_dict[s]])
                gradient[(s,t)] = total_density - self.constrained_val
        if return_violation:
            grad_arr = np.array(list(gradient.values()),copy=True)
            grad_arr[grad_arr < 0] = 0
            return gradient, np.linalg.norm(grad_arr, 2)
        else:
            return gradient
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                