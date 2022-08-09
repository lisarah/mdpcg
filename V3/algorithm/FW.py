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
    
def FW_dict(game, max_error, max_iterations, initial_density=None, verbose=True):
    
    y_list = [game.get_density() 
              if initial_density is None else initial_density]
    obj_list = [game.get_potential(y_list[0])]
    grad_list = [game.get_gradient(y_list[0])]
    k = 1
    err = max_error *2
    while k <= max_iterations and  abs(err) > max_error:
        y_k = y_list[-1]
        obj_list.append(game.get_potential(y_k))
        grad_list.append(game.get_gradient(y_k))
        V_k, pol_k = dp.value_iteration_dict(grad_list[-1], game.forward_P)
        sa_k, s_k = dp.density_retrieval(pol_k, game)
        step = 2 / (1+k)
        next_y = [{sa: step*d_t[sa] + (1-step)*y_t[sa] for sa in y_t.keys()}
                  for d_t, y_t in zip(sa_k, y_list[-1])]
        y_list.append(next_y)
        k += 1
        # compute error
        err = 0
        for g_t, y_1t, y_2t in zip(grad_list[-1], y_list[-1], y_list[-2]):
            err += sum([g_t[sa] * (y_1t[sa] - y_2t[sa]) for sa in g_t.keys()])
        if verbose:
            print(f'\r FW: error is {err} in {k} steps   ', end='')
    # print('')
    return y_list, obj_list
    

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
        print ("Frank wolfe ran out of iterations, error is", err)
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

