# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:08:09 2019

@author: Sarah Li
"""
import numpy as np
import algorithm.dynamic_programming as dp

def localFW(x0, p0, P, gradF, maxIterations = 5 ):
    it = 1;
    actions = 6;
    gradient = gradF(x0);
    xk  = x0;
    totalxK = np.zeros(actions);
    xHistory = [];
    xHistory.append(x0);
    while it <= maxIterations:
        step = 2./(1.+it);
        V, xNext = localSubproblem(gradient, p0, P);
        xk = (1. - step)* xk + step*xNext;
        gradient = gradF(xk);
        totalxK += 1.0*xk;
        xHistory.append(1.0*xk);
        it += 1;
    return xk, xHistory;

def localSubproblem(gradient, p0, P):
    actions= gradient.shape;
    xNext = np.zeros(actions);
    V = np.min(gradient);
    policy = np.argmin(gradient);
    xNext[int(policy)] = p0;
    return V, xNext;     
 
def FW(x0, p0, P, gradF, 
       isMax=False, 
       maxError = 1e-1, 
       returnLastGrad = False, 
       maxIterations = 5000, 
       returnHist = True):
    it = 1
    err= 1e13
    states, actions, time = x0.shape
    gradient = gradF(x0)
    xk = x0
    verbose = False
    if returnHist:
        xHistory = []
    else:
        xHistory = None;

    while it <= maxIterations and abs(err) >= maxError:
        step = 2./(1.+it);
        if verbose:
            print ("error: ", err)
        lastX =  1.0*xk;
        lastGrad = 1.0*gradient;
        V, xNext  = dp.value_iteration(gradient, p0, P,isMax);
        xk = (1. - step)* xk + step*xNext;
        gradient = gradF(xk);
        
        if returnHist:
#            totalxK += 1.0*xk;
#            xHistory.append(totalxK/it);
            xHistory.append(1.0*xk);
#        err = np.linalg.norm(lastGrad - gradient);
        err = np.sum(np.multiply(lastGrad,(lastX - xNext)));
        # print (f"error is {err}")
        
        it += 1;
    if it >= maxIterations:
        print ("ran out of iteraitons FW, error is", err);
    if returnLastGrad:
        return xk, xHistory, err;
    elif returnHist:
#        print ("FW approx: number of iterations ", it);
#        
        return xk, xHistory;
    else:
        if verbose:
            print (f"FW norm: {np.sum(xk)} ")
        return xk


def FW_inf(x0, p0, P, gradF, isMax=False, maxError = 1e-1, returnLastGrad = False, maxIterations = 100):
    it = 1;
    err= 1000.;
    states, actions = x0.shape;
    gradient = gradF(x0);
    xk  = x0;
    totalxK = np.zeros((states,actions));
    xHistory = [];
    xHistory.append(x0);
    dk = None;
    while it <= maxIterations and err >= maxError:
        step = 2./(1.+it);
#        print "error: ", err;
        lastGrad =  gradient;
        V, xNext  = dp.value_iteration(gradient, p0, P,isMax);
        xk = (1. - step)* xk + step*xNext;
        gradient = gradF(xk);
        totalxK += 1.0*xk;
#        xHistory.append(totalxK/it);
        xHistory.append(1.0*xk);
        err = np.linalg.norm(lastGrad - gradient);
        dk = xNext;
        it += 1;
    if returnLastGrad:
        return xk, xHistory, dk;
    else:
        return xk, xHistory;   

