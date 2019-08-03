# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:41:04 2018

@author: craba
"""

import Algorithms.dynamicProgramming as dp
import numpy as np
import numpy.linalg as la
from collections import namedtuple

GraphInfo = namedtuple("tsa", "time state action");

def projectGradient(R, C, D, Dc, C0, P, step, eps, realCost, ps, isFixed):
#    isFixed = True; # in this case, ps == None
    Wk = None;
    if isFixed == False: # ps is a zero vector
        Wk = np.multiply(ps, D) + Dc;
        print " is variable demand with upper bound";
    else:
        print " is Fixed demand";
    # 1: find random c0 to start with
    Ck = 1.0*C0;
    states,actions, time = R.shape;
    tsa = GraphInfo(time, states, actions);

    maxIter = 500;
    data = np.zeros(maxIter);
    stepSize = 1.0*step;
    loop = 0;
    runningObj = 0.5*realCost;
    polStar = None; VStar = 0.0; totalCost = 0.0; totalV = 0.0;
    while  abs((runningObj - realCost)/realCost) > eps and loop < maxIter: 
        if loop % 100 == 0:
            print "iteration: ", loop, "---------------------------";
            print "step Size: ", stepSize, "-------------------------";
            print "difference val = ",  abs((runningObj - realCost)/realCost);
        loop += 1;
        stepSize = step*loop**(-0.5);
        V, pol = dp.pgm_dynamicP(Ck, P);
        if isFixed:
            Cost = backwardPropagation_fixed(Ck, C, R, D, Dc, P, V, pol, tsa, stepSize, ps)
        else:
            Cost, WNext = backwardPropagation(Ck, C, R, D, Dc, P, V, pol, tsa, stepSize, ps, Wk);
        totalCost = totalCost + Cost;
        totalV = totalV + V;
        aveCost = 1.0*totalCost/loop;     
        aveV = 1.0*totalV/loop; 
        polStar = pol; 
        VStar = V;
        # phi part
#        CostB = np.divide(0.5*np.multiply(Cost - C, Cost - C), R); 
        runningObj = -0.5*np.sum(np.divide(np.multiply(aveCost - C, aveCost-C), R)) +aveV[:,0].dot(ps);
        # psi part
        if isFixed == False:
#            print "Is variable Demand in main pgm loop";
            CostA = 0.5*np.divide(np.multiply(V[:,0] - Dc, V[:,0] - Dc), D);
            runningObj += np.sum(CostA);
        data[loop-1] = runningObj;
        Ck = Cost;
        if isFixed == False:
            Wk = WNext;
        
        
    print "Final difference  ",   abs((runningObj - realCost)/realCost)
#    traj = dp.runPolicy(tsa.time, tsa.state, tsa.action, polStar, VStar, p0, P);
    return VStar, polStar, Ck, 0, data;
def backwardPropagation_fixed(Cold, C, R, D, Dc, P, V, pol, tsa, stepSize, ps):
    # pol : TxS -> A
    POpt = optimalMarkov(pol, P, tsa); # dimensions: T x S'' x S
    Delta = np.identity(tsa.state);
    step = 1.0*stepSize;
    # construct phi_inv(C_old)
    Cost = np.zeros((tsa.state, tsa.action, tsa.time));
#    Cost = Cold - step*np.divide(Cold - C, R);
    DeltaNew = np.zeros((tsa.state, tsa.state));
    for t in range(tsa.time-1):
        if t == 0:
            for s in range(tsa.state):
                Cost[s, int(pol[s,0]), 0] += step*ps[s];
                for a in range(tsa.action):
                    Cost[s, a, 0] += Cold[s,a, 0] - step*(Cold[s,a, 0] - C[s,a, 0])/R[s,a, 0];
                    Cost[s,a,0] = max(C[s, a,0], Cost[s, a, 0]);
                
        # part 1: get new D
        for s in range(tsa.state):
            for sp in range(tsa.state): 
                DeltaNew[sp, s] = sum([Delta[sp, spp]*POpt[t, s, spp] 
                                  for spp in range(tsa.state)]);                
        Delta = 1.0*DeltaNew;
        
        for s in range(tsa.state):
            aStar = int(pol[s,t+1]);
            Cost[s, aStar, t+1] += step*sum([
                    ps[sIter]*Delta[sIter, s] for sIter in range(tsa.state) ]);
            for a in range(tsa.action):
                Cp = Cold[s, a, t+1] ;
                Cost[s, a, t+1] += Cp - step*(Cp - C[s, a, t+1])/R[s,a, t+1];
                Cost[s, a, t+1] = max(C[s, a, t+1], Cost[s, a, t+1]); 
                
     
    return Cost;
def backwardPropagation(Cold, C, R, D, Dc, P, V, pol, tsa, stepSize, ps, Wk):
    # pol : TxS -> A
    POpt = optimalMarkov(pol, P, tsa); # dimensions: T x S'' x S
    Delta = np.identity(tsa.state);
    step = 1.0*stepSize;
    # construct phi_inv(C_old)
    Cost = np.zeros((tsa.state, tsa.action, tsa.time));
#    Cost = Cold - step*np.divide(Cold - C, R);
    DeltaNew = np.zeros((tsa.state, tsa.state));
    sigma = 1.0 *(V[:,0] < Wk);
    phis = -np.divide(Wk - Dc, D);
    WNext = np.zeros(tsa.state);     
    for t in range(tsa.time-1):
        if t == 0:
            for s in range(tsa.state):
                Cost[s, int(pol[s,0]), 0] += step*ps[s]*sigma[s];
                WNext = Wk - step*(ps[s] - phis[s]) + step*(1 - sigma[s])*ps[s];
                WNext = np.minimum(WNext, Dc);
                for a in range(tsa.action):
                    Cost[s, a, 0] += Cold[s,a, 0] - step*(Cold[s,a, 0] - C[s,a, 0])/R[s,a, 0];
                    Cost[s, a, 0] = max(C[s, a,0], Cost[s, a, 0]);
                
        # part 1: get new D
        for s in range(tsa.state):
            for sp in range(tsa.state): 
                DeltaNew[sp, s] = sum([Delta[sp, spp]*POpt[t, s, spp] 
                                  for spp in range(tsa.state)]);                
        Delta = 1.0*DeltaNew;
        
        for s in range(tsa.state):
            aStar = int(pol[s,t+1]);
            Cost[s, aStar, t+1] += step*sum([
                    sigma[sIter]*ps[sIter]*Delta[sIter, s] for sIter in range(tsa.state) ]);
            for a in range(tsa.action):
                Cp = Cold[s, a, t+1] ;
                Cost[s, a, t+1] += Cp - step*(Cp - C[s, a, t+1])/R[s,a, t+1];
                Cost[s, a, t+1] = max(C[s, a, t+1], Cost[s, a, t+1]); 
     
    return Cost, WNext;

def optimalMarkov(pol, P, tsa):
    # Time, state you end up in, state you start with = (t, s, s)
    POpt = np.zeros((tsa.time, tsa.state, tsa.state));
    for s in range(tsa.state):
        for t in range(tsa.time):
            aStar = int(pol[s,t]);
            POpt[t,:,s] = 1.0*P[:,s,aStar];
            
    return POpt;