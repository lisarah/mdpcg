# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:07:53 2021

@author: Sarah Li
"""
import numpy as np
import gameSolvers.cvx as cvx
import models.taxi_dynamics.manhattan_transition as m_dynamics
import models.taxi_dynamics.manhattan_neighbors as m_neighbors
import models.taxi_dynamics.visualization as visual

T = 15
borough = 'Manhattan'
# manhattan_game = cvx.cvx_solver(T, manhattan=True)
# driver_distribution = m_dynamics.uniform_initial_distribution(50000)
# manhattan_game.solve(driver_distribution, verbose=True, returnDual=False)

density_dict = {}
for zone_ind in m_neighbors.zone_neighbors.keys():
    density_dict[zone_ind] = np.random.rand()
visual.animate_borough(borough, density_dict)

zone_locations = visual.get_zone_locations(borough)
print(zone_locations)