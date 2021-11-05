# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:56:24 2021

Implementation of inexact projected gradient ascent. 

@author: Sarah Li
"""
from datetime import datetime
import numpy as np

def inexact_pga(tau_0, approx_gradient, step_size, max_iteration = 1000,
                epsilons = None, verbose = False):
    """ Perform inexact gradient ascent where projection into the positive
    quadrant is performed. 
    
    Args:
        tau_0: initial guess
        approx_gradient: function takes a tau value and epsilon to generate
          the appoximate gradient
        step_size: gradient descent step size
        max_iteration: maximum number of iterations for the algorithm
        epsilons: the accuracy achieved during each iteration
    Returns:
        tau_hist: tau value each iteration
        gradient_hist: the gradient value each iteration
    """
    if epsilons is None:
        # use the harmonic series as epsilons by default. 
        print('using harmonic series as epsilon in inexact projected gradient descent.')
        epsilons = [1/(k+1) for k in range(max_iteration)]
    
    tau_hist = [tau_0]
    gradient_hist = []
    t_start = datetime.now()
    for t in range(max_iteration):
        if verbose and t % 100 == 0:
            t_end = datetime.now()
            t_diff = (t_end - t_start).total_seconds()
            print(f'inexact pga ---- iteration = {t}, time = {t_diff} ----')
            print(np.linalg.norm(tau_hist[-1], 2))
        gradient = approx_gradient(tau_hist[-1], epsilons[t])
        gradient_hist.append(gradient)
        tau_next = tau_hist[-1] + step_size * gradient
        tau_next[tau_next < 0] = 0
        tau_hist.append(tau_next)

    return tau_hist, gradient_hist

def fast_inexact_pga(tau_0, approx_gradient, step_size, max_iteration = 1000,
                epsilons = None, verbose = False, lipschitz = 1):
    """ Perform inexact gradient ascent where projection into the positive
    quadrant is performed. 
    
    Args:
        tau_0: initial guess
        approx_gradient: function takes a tau value and epsilon to generate
          the appoximate gradient
        step_size: gradient descent step size
        max_iteration: maximum number of iterations for the algorithm
        epsilons: the accuracy achieved during each iteration
    Returns:
        tau_hist: tau value each iteration
        gradient_hist: the gradient value each iteration
    """
    if epsilons is None:
        # use the harmonic series as epsilons by default. 
        epsilons = [1/(k+1) for k in range(max_iteration)]
    
    tau_hist = [tau_0]
    gradient_hist = []
    t_start = datetime.now()
    w_t = 0
    for t in range(max_iteration):
        if verbose and t % 100 == 0:
            t_end = datetime.now()
            t_diff = (t_end - t_start).total_seconds()
            print(f'inexact pga ---- iteration = {t}, time = {t_diff} ----')
            print(np.linalg.norm(tau_hist[-1], 2))
        gradient = approx_gradient(tau_hist[-1], epsilons[t])
        gradient_hist.append(gradient)
        tau_next = tau_hist[-1] + gradient * step_size
        tau_next[tau_next < 0] = 0
        
        alpha_t =  1/(t+1) # (t + 1) / 2 #
        # A_t = (t+1) * (t+2) / 4/ lipschitz
        # gradient[gradient < 0 ] = 0
        w_t += alpha_t * gradient # / lipschitz
        non_zero_w_t  = w_t * 1
        non_zero_w_t[non_zero_w_t < 0] = 0
        
        
        
        frac = 2 / (t + 3)
        tau_next = (1 - frac) * tau_next + frac * non_zero_w_t 
        tau_hist.append(tau_next)

    return tau_hist, gradient_hist
















