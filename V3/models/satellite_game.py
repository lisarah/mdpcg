# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:36:48 2021

Satellite game contains the costs, states, actions of a satellite game. 

States: each state corresponds to an initial orbit where there's a satellite.
Actions: each action corresponds to a final orbit where there should be a
    satellite.
Costs: The hohnman trasfer velocity to get from initial orbit (state) to final
 orbit (action).
 
@author: Sarah Li
"""
import util.orbital_mechanics as om

class satellite_game:
    def __init__(self, r_init_set, r_final_set):
        """ Initialize satellite game.
        
        Args:
            r_init_set: initial radii list, assume 1 satellite per orbit.
            r_final_set: final radii list, assume 1 satellite per orbit.
        """
        self.cost= []
        self.states={}
        self.actions = {}
        state_ind = 0
        action_ind = 0
        for r_init in r_init_set:
            self.cost.append([])
            self.states[state_ind] = r_init
            for r_final in r_final_set:
                if state_ind == 0:
                    self.actions{action_ind} = r_final 
                action_ind += 1
                self.cost[-1].append(om.hohnman_transfer(r_init, r_final))
            self.state_ind += 1

                

