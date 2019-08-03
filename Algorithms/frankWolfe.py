# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:54:43 2018

@author: craba
"""
import numpy as np
import Algorithms.dynamicProgramming as dp
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
 
def FW(x0, p0, P, gradF, isMax=False, maxError = 1e-1, returnLastGrad = False, maxIterations = 5000):
    it = 1;
    err= 1000.;
    states, actions, time = x0.shape;
    gradient = gradF(x0);
    xk  = x0;
    totalxK = np.zeros((states,actions,time));
    xHistory = [];
    xHistory.append(x0);
    dk = None;
    while it <= maxIterations and err >= maxError:
        step = 2./(1.+it);
#        print "error: ", err;
        lastX =  1.0*xk;
        lastGrad = 1.0*gradient;
        V, xNext  = subproblem(gradient, p0, P,isMax);
        xk = (1. - step)* xk + step*xNext;
        gradient = gradF(xk);
        totalxK += 1.0*xk;
#        xHistory.append(totalxK/it);
        xHistory.append(1.0*xk);
#        err = np.linalg.norm(lastGrad - gradient);
        err = -np.sum(np.multiply(lastGrad,(lastX - xNext)));
#        print "error is ", err;
        dk = xNext;
        it += 1;
    if it >= maxIterations:
        print "ran out of iteraitons FW, error is", err;
    else:
        print " took iterations ", it;
    if returnLastGrad:
        return xk, xHistory, err;
    else:
        return xk, xHistory;

def subproblem(gradient, p0, P, isMax = False):
    states, actions, time = gradient.shape;
    V = np.zeros((states, time));
    policy = np.zeros((states, time)); # pi_t(state) = action;
    trajectory = np.zeros((states,time));
    xNext = np.zeros((states,actions,time));

    # construct optimal value function and policy
    for tIter in range(time):
        t = time-1-tIter;   
        cCurrent =gradient[:,:,t]; 
        if t == time-1:       
            if isMax:
                V[:,t] = np.max(cCurrent, axis = 1);
                policy[:,t] = np.argmax(cCurrent, axis=1);
            else:                 
                V[:,t] = np.min(cCurrent, axis = 1);
                policy[:,t] = np.argmin(cCurrent, axis=1);
        else:
            # solve Bellman operators
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P,Vt);
            if isMax:
                V[:,t] = np.max(obj, axis=1);
                policy[:,t] = np.argmax(obj, axis=1);
            else:
                V[:,t] = np.min(obj, axis=1);
                policy[:,t] = np.argmin(obj, axis=1);

    for t in range(time):
        # construct next trajectory
        if t == 0:
            traj = 1.0*p0;
        else:
            traj = trajectory[:,t-1];
        # construct y
        pol = policy[:,t];
        x = np.zeros((states,actions));

        for s in range(states):
            x[s,int(pol[s])] = traj[s];
        xNext[:,:,t] = 1.0*x;
        trajectory[:,t] =  np.einsum('ijk,jk',P,x);

    return V, xNext; 
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
        V, xNext  = subproblem(gradient, p0, P,isMax);
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
def FW_fixedDemand(x0,z0, demand, P, gradF, maxError = 1e-1):
    maxIterations = 500;
    it = 1;
    err= 1000;
    states, actions, time = x0.shape;
    gx,gz = gradF(x0,z0);
    xk  = x0; zk = z0;
    xHistory = [];
    xHistory.append(x0);
    while it <= maxIterations and err >= maxError:
        step = 2./(1+it);
        lastGx =  gx; lastGz = gz;
        V, xNext,zNext, pol = dp.subproblem_fixedDemand(lastGx, lastGz, demand, P);
        xk = (1. - step)* xk + step*xNext;
        zk = (1. - step)* zk + step*zNext;
        gx,gz = gradF(xk,zk);
        err = np.linalg.norm(lastGx - gx) + np.linalg.norm(lastGz - gz);
        xHistory.append(xk);
        it += 1;
    print " ------------ FW_fixedDemand summary -----------";
    print "number of iterations = ", it;
    print "total error in cost function = ", err;
    return xk, zk, xHistory;

