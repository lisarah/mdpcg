# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:13:44 2021

Solve the satellite game using an auction algorithm

@author: Sarah Li
"""
import algorithm.auction as auction
import models.satellite_game as s_game
import gameSolvers.cvx_satellite as cvx

r_init = [700, 800, 900, 950, 990] # km, 1 satellite per orbit
r_final = [900, 1000, 1050, 1100, 1150] # km, 1 satellite per orbit
satellite_game = s_game.satellite_game(r_init, r_final)

# # auction algorithm
initial_prices = [0, 0, 0, 0, 0]
auction_assignment = auction.auction(satellite_game, initial_prices)
auction_distribution = auction.get_bidder_distribution(auction_assignment, 
                                                       len(r_init), 
                                                       len(r_final))

print(f'auction minimized distribution \n {auction_distribution}')
print(f'auction final solution: {satellite_game.get_objective(auction_distribution)}')

# CVX solution
cvx_solver = cvx.cvx_satellite(r_init, r_final)
cvx_distribution = cvx_solver.solve()
print(f'cvx final solution: {satellite_game.get_objective(cvx_distribution)}')