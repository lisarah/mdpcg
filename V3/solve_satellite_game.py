# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:13:44 2021

Solve the satellite game using an auction algorithm

@author: Sarah Li
"""
import algorithm.auction as auction
import models.satellite_game as s_game
import gameSolvers.cvx_satellite as cvx

r_init = [700, 800] # km, 1 satellite per orbit
r_final = [900, 1000] # km, 1 satellite per orbit
satellite_game = s_game.satellite_game(r_init, r_final)

# auction algorithm
initial_prices = [0, 1]
optimal_assignment = auction.auction(satellite_game, initial_prices)

for action, state in optimal_assignment.items():
    print(f'initial orbit {satellite_game.states[state]} -> '
          f'final orbit {satellite_game.actions[action]}')


# CVX solution
cvx_solver = cvx.cvx_satellite(r_init, r_final)
cvx_solver.solve()