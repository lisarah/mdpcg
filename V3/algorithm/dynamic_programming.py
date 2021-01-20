# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:18:30 2021

Dynamic programming methods: 
    - value iteration for both maximizing and minimizing objective.
    
@author: Sarah Li
"""
import numpy as np


def value_iteration(cost, p0, P, isMax = False):
    """ Value iteration with max/min objectives for a finite time horizon, total
    cost MDP.
    
    Inputs:
        cost: np array with shape (S, A, T).
        p0: initial probabilitiy distribution, np array with length S
        P: transition kernel, np array with shape (S, S, A, T), P[s,r,a,t] is
          the probability of transition from state r to state s by taking 
          action a at time t. 
    Returns:
        V: values of each state, np array with length S.
        x_next: the optimal population distribution, np array with shape 
        (S,A,T).
    """
    S, A, T = cost.shape;
    V = np.zeros((S, T));
    policy = np.zeros((S, T)); # pi_t(state) = action;
    trajectory = np.zeros((S,T));
    xNext = np.zeros((S,A,T));

    # construct optimal value function and policy
    for tIter in range(T):
        t = T-1-tIter;   
        cCurrent =cost[:,:,t]; 
        if t == T-1:       
            if isMax:
                V[:,t] = np.max(cCurrent, axis = 1);
                policy[:,t] = np.argmax(cCurrent, axis=1);
            else:                 
                V[:,t] = np.min(cCurrent, axis = 1);
                policy[:,t] = np.argmin(cCurrent, axis=1);
        else:
            # solve the Bellman operator
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P,Vt);
            if isMax:
                V[:,t] = np.max(obj, axis=1);
                policy[:,t] = np.argmax(obj, axis=1);
            else:
                V[:,t] = np.min(obj, axis=1);
                policy[:,t] = np.argmin(obj, axis=1);

    # construct the optimal trajectory corresponding to the Bellman operator.
    for t in range(T):
        if t == 0:
            traj = 1.0*p0;
        else:
            traj = trajectory[:,t-1];
        # construct y
        pol = policy[:,t];
        x = np.zeros((S,A));

        for s in range(S):
            x[s,int(pol[s])] = traj[s];
        xNext[:,:,t] = 1.0*x;
        trajectory[:,t] =  np.einsum('ijk,jk',P,x);

    return V, xNext; 