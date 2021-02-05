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
class congestion_parameters:
    def __init__(self):
        self.tau = 27.  # $/hr
        self.vel = 8. # mph
        self.fuel = 28. # $/gal
        self.fuelEff = 20. # mi/gal
        self.rate = 6. # $/mi
        # cost constant for going somewhere else
        self.k = self.tau/self.vel + self.fuel/self.fuelEff
        


def avg_trip_distance(s):
    """TODO not implemented yet.
    Return the average distance travelled for trips starting in state s.
    """
    return 10
       
def congestion_cost(ride_demand, T ,S, A, epsilon = 0):
    """ Generate the congestion cost vector ell_{tsa}.
    Each ell_{tsa} = R_{tsa} y_{tsa} + C_{tsa}
    
    Input:
        rider_demand: a list of length S with the rider demand in each state
        T: total time steps
        S; total number of states
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
            C[s, a, t] = (params.k - params.rate) * avg_trip_distance(s) 
            R[s, a, t] = params.tau / ride_demand[s]
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
                C[s, a, t] = params.k*haversine(s_latlon, n_latlon) 
                R[s, a, t] = epsilon # indedpendent of distance
                 
    return R, C

