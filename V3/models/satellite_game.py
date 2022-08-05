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
    def __init__(self, r_init_set, r_final_set, collision_cost = 0.5):
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
                if r_final == r_init:
                    self.cost[-1].append(0)
                else:
                    self.cost[-1].append(om.hohnman_transfer(r_init, r_final))
            state_ind += 1
        self.cost = np.array(self.cost)
        
        self.collision_set = self.get_collisions()
        
    def get_collisions(self):
        """ Given derive the collision sets. 
        A collision occurs between (s, a) and (s', a') whenever 
            - s  <s' and a > a' or
            - s > s' and a < a'
        The collision set includes each collision between (s, a) and (s', a')
        only once. 
        
        Returns:
            collisions: a list of collision orbits
        """
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
    
    def get_cost(self, y_sa, s, a):
        """ Return the congested cost of (s, a) at current distribution.
        
        Args:
            y_sa: [S] x [A] np array of current distribution
            s: the state of requested cost
            a: the action of requested cost
        Returns:
            cost: the cost of (s,a) at distribution y_sa.
        """
        cost =  self.cost[s,a]
        for collision in self.collision_set:
            if s == collision[0] and a == collision[1]: 
                cost += self.k * y_sa[collision[2], collision[3]]
            elif s == collision[2] and a == collision[3]:
                cost += self.k * y_sa[collision[0], collision[1]]
        return cost
    
    def get_objective(self, y_sa):
        """ Return objective value at current y_sa distribution.
        
        Args:
            y_sa: state-action distribution.
        Returns:
            obj: objective value.
        """
        obj = 0
        for s_ind in range(len(self.states)):
            for a_ind in range(len(self.actions)):
                obj += y_sa[s_ind, a_ind] * self.get_cost(y_sa, s_ind, a_ind)
                
        return obj
    
    def set_objective(self, y_sa, w):
        """ Set the objective function for cvx in terms of cvx variables.
        
        Args:
            y_sa: cvx variable matrix of current state-action distribution.
            w: cvx auxiliary variable for relaxing the problem.
        Returns:
            objective: objective function in terms of cvx variables.
        """
        objective = sum([ sum([ 
            y_sa[s,a] * self.cost[s,a] 
            for a in self.actions.values()]) for s in self.states.values()])
        
        for c in self.collision_set:
            # collision cost
            x = y_sa[c[0], c[1]]
            y = y_sa[c[2], c[3]]
            # this is a convex relaxation using AM-GM: \sqrt(ab) <= 0.5*(a + b)
            objective += 0.5 * self.k * (cvx.square(x) + cvx.square(y))
        return objective 
