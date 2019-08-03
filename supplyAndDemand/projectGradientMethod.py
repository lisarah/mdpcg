# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:41:04 2018

@author: craba
"""

import dynamicProgramming as dp
import numpy as np
import numpy.linalg as la
from collections import namedtuple


GraphInfo = namedtuple("tsa", "time state action");

def projectGradient(R, C, p0, C0, P, stepSize, eps, tsa, realCost):
    # 1: find random c0 to start with
    Cost = C0;
    C0 = np.zeros((tsa.state,tsa.action, tsa.time));
    maxIter = 1000;
    data = np.zeros(maxIter);
   
    loop = 0;
    totalCost = 0; totalV = 0;
    runningAverageObj = 10;
    while  abs(runningAverageObj - 218094.34132) > eps and loop < maxIter: 
        if loop % 100 == 0:
            print "iteration: ", loop, "---------------------------";
            print "difference val = ", abs(runningAverageObj - 218094.34132);
        loop += 1;
        stepSize = stepSize*loop**(-0.5);
        C0 = Cost;
        V, pol = dp.dynamicP(C0, P, p0);
        Cost = backwardPropagation(C0, C, R, P, V, pol, tsa, p0, stepSize);
        totalCost = totalCost + Cost;
        totalV = totalV + V;
        aveCost = 1.0*totalCost/loop;     
        aveV = 1.0*totalV/loop;
        runningAverageObj = -np.sum(0.5*np.divide(np.multiply(aveCost - C, aveCost-C), R)) \
            +aveV[:,0].dot(p0);
        data[loop-1] = runningAverageObj;
        
    print "Final difference  ",  abs(runningAverageObj - 218094.34132) 
    traj = dp.runPolicy(tsa.time, tsa.state, tsa.action, pol, V, p0, P);
    return V, pol, Cost, traj, data;

def backwardPropagation(Cold, C, R, P, V, pol, tsa, p0, stepSize):
    # pol : TxS -> A
    POpt = optimalMarkov(pol, P, tsa); # dimensions: T x S'' x S
    Delta = np.identity(tsa.state);
    Cost = np.zeros((tsa.state, tsa.action, tsa.time));
    DeltaNew = np.zeros((tsa.state, tsa.state));
    step = 1.0*stepSize;
    for t in range(tsa.time-1):
        if t == 0:
            for s in range(tsa.state):
                for a in range(tsa.action):
                    Cost[s, a, 0]= Cold[s,a, 0] - step*(Cold[s,a, 0] - C[s,a, 0])/R[s,a, 0];  
                Cost[s, int(pol[s,0]), 0] += step*p0[s];
                
        # part 1: get new D
        for s in range(tsa.state):
            for sp in range(tsa.state): 
                DeltaNew[sp, s] = sum([Delta[sp, spp]*POpt[t, s, spp] 
                                  for spp in range(tsa.state)]);                
        Delta = DeltaNew;
        
        for s in range(tsa.state):
            aStar = int(pol[s,t+1]);
            Cost[s, aStar, t+1] += step*sum([
                    p0[sIter]*Delta[sIter, s] for sIter in range(tsa.state) ])

            for a in range(tsa.action):
                Cp = Cold[s, a, t+1] ;
                Cost[s, a, t+1] += Cp - step*(Cp - C[s, a, t+1])/R[s,a, t+1];
                Cost[s, a, t+1] = max(C[s, a, t+1], Cost[s, a, t+1]); 
    
    return Cost;

def optimalMarkov(pol, P, tsa):
    # Time, state you end up in, state you start with = (t, s, s)
    POpt = np.zeros((tsa.time, tsa.state, tsa.state));
    for s in range(tsa.state):
        for t in range(tsa.time):
            aStar = int(pol[s,t]);
            POpt[t,:,s] = 1.0*P[:,s,aStar];
            
    return POpt;