# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:51:33 2019

@author: craba
"""
#powell's method
from mystic.solvers import fmin_powell, fmin
# rosenbrock function
from mystic.models import rosen
#tools
from mystic.monitors import VerboseMonitor

if __name__ == '__main__':
    print ("powell's method")
    print ("===============")
    x0 = [0.3, 1.2, 0.4];
    # quadratic objective
    def quadObj(x):
        return abs(x[0]-2.) + abs(x[1] - 3.) + abs(x[2] - 4.);
    # define constraints 
    def constraints(x):
        # constrain the last x_i to be the same value as the first x_i
        x[-1] = x[0];
        return x;
    
    # configure monitor
    stepmon = VerboseMonitor(1);
    
    # use Powell's method to minimize the Rosenbrock function UNCONSTRAINED
    solution = fmin_powell(quadObj,x0, itermon=stepmon);
    print (solution);
    
    print ("regular fmin method")
    print ("===================")
    x0 = [0.3, 1.2, 0.4];
    # configure monitor
    stepmon = VerboseMonitor(1);
    
    # use Powell's method to minimize the Rosenbrock function constrained
    solution = fmin(quadObj,x0, constraints = constraints, itermon=stepmon);
    print (solution);