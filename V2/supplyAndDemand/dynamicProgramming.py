# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 15:40:26 2018

@author: craba
"""
import numpy as np;

def iterativeDP(states, actions, time, numPlayers, p0, 
                R, C, P,
                hasToll = False, 
                toll = None, 
                tollState = None):
    yt = np.zeros((states, actions, time));
    totalVal = 0;
    VTotal = np.zeros((states, time));  
    for p in range(numPlayers):
    #    startPt = residentialList[p%residentialNum];
    #    startCdn = np.zeros((seattleGraph.number_of_nodes()));
    #    startCdn[startPt] = 1.0;
        startCdn = p0/numPlayers;
        V, valNext, yNext =  instantDP(R,
                                       C,
                                       P,
                                       yt, 
                                       startCdn, 
                                       hasToll = hasToll,
                                       toll = toll,
                                       tollState = tollState);
        yt += 1.0*yNext;
        totalVal += 1.0*valNext;   
        VTotal += V;
        
    return V, totalVal, yt;
    
def instantDP(R, C, P,yt, p0, hasToll =False, toll = None, tollState = None):
    states,actions,time = R.shape;
    V = np.zeros((states, time));
    policy = np.zeros((states, time)); # pi_t(state) = action;
    trajectory = np.zeros((states,time));
    yNext = np.zeros((states,actions,time));
    # construct optimal value function and policy
    for tIter in range(time):
        t = time-1-tIter;   
        if t == time-1:
            cCurrent =-np.multiply(R[:,:,t], yt[:,:,t]) + C[:,:,t];
            if hasToll:    
                if (abs(toll[t])> 0):
                    cCurrent[tollState,:] += toll[t];
                    
            V[:,t] = np.max(cCurrent, axis = 1);
            pol = np.argmax(cCurrent, axis=1);
            policy[:,t] = pol;
        else:
            cCurrent =-np.multiply(R[:,:,t], yt[:,:,t]) + C[:,:,t];
            if hasToll:    
                if (abs(toll[t])> 0):
                    cCurrent[tollState,:] += toll[t];
            # solve Bellman operators
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P,Vt);
            V[:,t] = np.max(obj, axis=1);
            pol = np.argmax(obj, axis=1);
            policy[:,t] = pol;

    for t in range(time):
        # construct next trajectory
        if t == 0:
            traj = 1.0*p0;
        else:
            traj = trajectory[:,t-1];
        # construct y
        pol = policy[:,t];
        y = np.zeros((states,actions));

        for s in range(states):
            y[s,int(pol[s])] = traj[s];
        yNext[:,:,t] = 1.0*y;
        trajectory[:,t] =  np.einsum('ijk,jk',P,y);

            
    val = sum([p0[state]*V[state,0] for state in range(states)]) \
          + 0.5*sum([sum([sum([R[state,a,t]*yNext[state,a,t]*yNext[state,a,t]
                for t in range(time)])  for state in range(states)]) for a in range(actions)]);

    return V, val, yNext;
 
def dynamicPLinearCost(R, C, P,yt, p0, hasToll =False, toll = None, tollState = None):
    states,actions,time = R.shape;
    V = np.zeros((states, time));
    policy = np.zeros((states, time)); # pi_t(state) = action;
    trajectory = np.zeros((states,time));
    # construct optimal value function and policy
    for tIter in range(time-1):
        t = time-1-tIter;   
        print "----------------------t = ", t," -----------------";
        if t == time-1:
            cCurrent =-np.multiply(R[:,:,t], yt[:,:,t]) + C[:,:,t];
            if hasToll:    
                cCurrent = cCurrent - toll[t];
#            print cCurrent.shape;
            V[:,t] = np.max(cCurrent, axis = 1);
#            print cCurrent;
#            print V[:,t]
        else:
            cCurrent =-np.multiply(R[:,:,t], yt[:,:,t]) + C[:,:,t];
            if hasToll:    
                cCurrent = cCurrent - toll[t];
            # solve Bellman operators
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P,Vt);
            V[:,t] = np.min(obj, axis=1);
#            for s in range(states):
#                for a in range(actions):
#                    if obj[s,a]+ 1e-8 >=V[:,t]:
#                        a 
            pol = np.argmin(obj, axis=1);
            policy[:,t+1] = pol;
#            print obj
#            print V[:,t];
#            print pol;
            

            
    for t in range(time):
        # construct next trajectory
        if t == 0:
#            bestState = np.argmax(V[:,0]);
#            trajectory[bestState,t] = 1.0;
            trajectory[:,t] = p0;
        else:
            # construct y
            pol = policy[:,t];
#            print pol;
            y = np.zeros((states,actions));
            traj = trajectory[:,t-1];
            for s in range(states):
                y[s,int(pol[s])] = traj[s];
            trajectory[:,t] =  np.einsum('ijk,jk',P,y);
    
    print -sum([p0[state]*V[state,0] for state in range(states)])-0.5*sum([sum([R[state,0,t]*trajectory[state,t]*trajectory[state,t] for t in range(time)])  for state in range(states)]);
    return V, trajectory;

    
def dynamicP(c, P, p0, minimize = True):
    states,actions,time = c.shape;
    V = np.zeros((states, time));
    policy = np.zeros((states, time)); # pi_t(state) = action;
#    trajectory = np.zeros((states,actions, time));
    # construct optimal value function and policy

    for tIter in range(time):
        t = time-1-tIter;    
        cCurrent = c[:,:, t];
        obj = cCurrent;
        if t != time - 1:
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P,Vt);
            
        if minimize: 
            V[:,t] = np.min(obj, axis=1);
            pol = np.argmin(obj, axis =1);
        else:
            V[:,t] = np.max(obj, axis=1);
            pol = np.argmax(obj, axis =1);
                
        policy[:,t] = pol;   
    return V, policy;   

def runPolicy(time, states, actions, policy, V, p0, P ):
    trajectory = np.zeros((states, actions, time))
    # propagate the population density using optimal trajectory        
    for t in range(time):
        pol = policy[:,t];
        if t == 0:
            population= p0;
        else:
            population = np.einsum('ijk,jk',P,trajectory[:,:,t-1]);
        for s in range(states):
            trajectory[:, int(pol[s]), t] = population[s];

    return trajectory;        
