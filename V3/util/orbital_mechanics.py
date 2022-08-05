# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:18:02 2021

Orbital mechanic calculations for LEO satellites. 

@author: Sarah Li
"""
import numpy as np


_MU = 398600.442 # km^3/s^2. Standard gravitational parameter for Earth
_RE = 6371 # km. Earth's radius
def hohnman_transfer(r_init, r_final):
    """ Compute the velocity changes required to change orbit from A to B.
    See https://en.wikipedia.org/wiki/Hohmann_transfer_orbit. 
    """
    r_transfer = 0.5 * (r_init + r_final)
    
    v_1 = np.sqrt(_MU * (2 / r_init - 1 / r_init)) # km/s.
    v_transfer_init = np.sqrt(_MU * (2 / r_init - 1 / r_transfer)) # km/s.
    v_transfer_end = np.sqrt(_MU * (2 / r_final - 1 / r_transfer)) # km/s.
    v_2 = np.sqrt(_MU * (2 / r_final - 1 / r_final)) # km/s.
    
    delta_1 = abs(v_1 - v_transfer_init)
    delta_2 = abs(v_2 - v_transfer_end)
    
    return delta_1 + delta_2
def test_hohnman_transfer():
    """ r_initial  = 7000 km, r_final = 20000km 
        requires total delta v = 2.888 km/s.
    """
    r_init = 7000
    r_final = 20000
    true_delta_v = 2.888
    returned_delta_v = hohnman_transfer(r_init, r_final)
    if np.testing.assert_approx_equal(returned_delta_v, true_delta_v, 3):
        print (f'hohnman transfer failed. Answer: {true_delta_v} km/s,'
               f'function returned: {hohnman_transfer(r_init, r_final)}')
    