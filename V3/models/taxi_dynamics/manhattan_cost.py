# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:37:56 2021

Calculates the congestion costs using the cost model from
https://arxiv.org/abs/1903.00747

Code requires Haversine
conda install -c conda-forge haversine
 
@author: Sarah Li
"""
import numpy as np
import models.taxi_dynamics.manhattan_neighbors as manhattan
import models.taxi_dynamics.visualization as geography
from haversine import haversine
import pandas as pd

_km_to_mi = 0.621371 

class congestion_parameters:
    def __init__(self):
        self.base_rate = 2.55  + 4.2 # $ base rate plus 12 minutes of ride
        self.rate_mi = 1.75  # $/mi
        self.tau = 15  # $/hr 
        self.vel = 12. # mph
        self.fuel = 2.8 # $/gal
        self.fuelEff = 28. # mi/gal
        # cost constant for travelling
        self.k = self.tau/self.vel + self.fuel/self.fuelEff #  
        
        
def demand_rate(file, Timesteps, States):
    """ Extract demand rate per state per time step from file.
    
    Args:
        file: name of file
        Timesteps: number of time steps.
        States: number of states.
    Returns:
        demand_rate: a 2D list where the [t][s]^th element is the demand at
          state s and time t.
    """
    demand_array = pd.read_csv(file, header=0).values
    demand_rate = []
    for t in range(Timesteps):
        demand_rate.append([sum(demand_array[s, t*States:(t+1)*States]) 
                            for s in range(States)])
    return demand_rate
# def avg_trip_distance():
#     """TODO not implemented yet.
#     Return the average distance travelled for trips starting in state s.
#     """
#     pd.read_csv(file, header=0).values
#     return 10
       
def congestion_cost(ride_demand, T ,S, A, avg_trip_dist, epsilon = 0):
    """ Generate the congestion cost vector ell_{tsa}.
    Each ell_{tsa} = R_{tsa} y_{tsa} + C_{tsa}
    
    Input:
        rider_demand: a list of length S with the rider demand in each state
        T: total time steps
        S: total number of states
        A: number of actions
        avg_trip_dist: a list of average trip distance, indexed by state ind.
    Output:
        R: linear part of ell
        C: constant part of ell
    """
    C = np.zeros((S, A , T))
    R = np.zeros((S, A , T))
    params = congestion_parameters()
    state_ind  = manhattan.zone_to_state(manhattan.zone_neighbors)
    zone_ind = {z_ind: s_ind for s_ind, z_ind in state_ind.items()}
    zone_geography = geography.get_zone_locations('Manhattan')
    for t in range(T):
        for s in range(S):
            a = A - 1  # picking up riders
            m_pick_up = (params.base_rate + params.rate_mi * 
                         avg_trip_dist[t][s] * _km_to_mi ) 
            m_pick_up = max([7, m_pick_up])
            C[s, a, t] =  (-m_pick_up  + 
                           params.k * avg_trip_dist[t][s] * _km_to_mi)
            R[s, a, t] = m_pick_up / (3 * ride_demand[t][s] / 31)
            neighbors = manhattan.STATE_NEIGHBORS[s]
            N_neighbors = len(neighbors)
            for a in range(A - 1): # going to neighbor
                if a < N_neighbors:
                    neighbor = manhattan.STATE_NEIGHBORS[s][a]
                else:
                    neighbor = manhattan.STATE_NEIGHBORS[s][N_neighbors - 1]
                s_latlon = zone_geography[zone_ind[s]]
                n_latlon = zone_geography[zone_ind[neighbor]]
                # haversine returns distance between two lat-lon tuples in km.
                # 0.621371 converts km to mi.
                C[s, a, t] = (params.k * haversine(s_latlon, n_latlon) * 
                              _km_to_mi) 
                R[s, a, t] = epsilon # indedpendent of distance
                 
    return R, C

