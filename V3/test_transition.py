# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:34:49 2021

@author: Sarah Li
"""
import numpy as np
import models.taxi_dynamics.manhattan_transition as dynamics
import pandas as pd
import models.taxi_dynamics.visualization as visual
import models.taxi_dynamics.manhattan_cost as m_cost


T = 15

time_intervals = np.linspace(0,15, 16)
new_kernel = dynamics.extract_kernel("transition_kernel.csv", 15, 63)
count_array = pd.read_csv("count_kernel.csv", header=0).values

demand_analysis = np.zeros((63, 63, T))
for t in range(T):
    demand_analysis[:,:, t] = new_kernel[t]
# visual.plot_borough_progress('Manhattan', demand_analysis, [0, int(T/2), T-1])

weighted_average = pd.read_csv("weighted_average.csv", header=0).values

demand_rate = m_cost.demand_rate('count_kernel.csv', T, 63)
