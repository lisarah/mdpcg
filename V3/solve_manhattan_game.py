# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:07:53 2021

@author: Sarah Li
"""
import gameSolvers.cvx as cvx
import models.taxi_dynamics.manhattan_transition as m_dynamics

T = 15
manhattan_game = cvx.cvx_solver(T, manhattan=True)
driver_distribution = m_dynamics.uniform_initial_distribution(50000)
manhattan_game.solve(driver_distribution, verbose=True, returnDual=False)
