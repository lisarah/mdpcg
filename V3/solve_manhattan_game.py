# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:07:53 2021

@author: Sarah Li
"""
import models.taxi_dynamics.manhattan_neighbors as manhattan

T = 20
P = manhattan.manhattan_transition_kernel(T, 0.1)
manhattan.test_manhattan_transition_kernel(P)