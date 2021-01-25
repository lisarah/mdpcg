# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:36:48 2021

Satellite game. 

@author: Sarah Li
"""
import cvxpy as cvx
import numpy as np
import util.orbital_mechanics as om


class satellite_game:
    """ Satellite game contains the costs, states, actions of a satellite game. 
    
        States: each state corresponds to an initial orbit where there's a 
        satellite.
        Actions: each action corresponds to a final orbit where there should be
        a satellite.
        Costs: The hohnman trasfer velocity to get from initial orbit (state)
        to final orbit (action).
    """
    def __init__(self, r_init_set, r_final_set, collision_cost = 0.02):
        """ Initialize satellite game.
        
        Args:
            r_init_set: initial radii list, assume 1 satellite per orbit.
            r_final_set: final radii list, assume 1 satellite per orbit.
        """
        self.cost= []
        self.states={}
        self.actions = {}
        self.k = collision_cost
        state_ind = 0
        action_ind = 0
        for r_init in r_init_set:
            self.cost.append([])
            self.states[r_init] = state_ind
            for r_final in r_final_set:
                if state_ind == 0:
                    self.actions[r_final] = action_ind
                    action_ind += 1
                self.cost[-1].append(om.hohnman_transfer(r_init, r_final))
            state_ind += 1
        self.cost = np.array(self.cost)
        
        self.collision_set = self.get_collisions()
        
    def get_collisions(self):
        sorted_states = sorted(self.states.keys())
        sorted_actions = sorted(self.actions.keys())
        collisions = []
        for state_1 in sorted_states:
            prev_states = sorted_states[:sorted_states.index(state_1)]
            for action_1 in sorted_actions:
                a_1_ind = sorted_actions.index(action_1)
                next_actions = sorted_actions[a_1_ind + 1:]
                # print(next_actions)
                for state_2 in prev_states:
                    # print('state', state_2)
                    for action_2 in next_actions:
                        # print('action', action_2)
                        collisions.append(
                            (self.states[state_1], self.actions[action_1], 
                             self.states[state_2], self.actions[action_2]))
                        # print(f'added collision {collisions[-1]}')
        # print(f'collisions: {collisions}')
        return collisions
    
    def set_objective(self, y_sa, w):
        objective = sum([ sum([ 
            y_sa[s,a] * self.cost[s,a] 
            for a in self.actions.values()]) for s in self.states.values()])
        
        for c in self.collision_set:
            # collision cost
            x = y_sa[c[0], c[1]]
            y = y_sa[c[2], c[3]]
            objective += 0.5 * self.k * (cvx.square(x) + cvx.square(y))
        return objective 
