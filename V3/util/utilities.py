# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 09:40:58 2018

@author: craba
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


# specify which graph we are using for the mdp problem
Constraint = namedtuple("constraints", "index value upperBound");
# truncate array values to something more readable
def truncate(tau, thres = 5e-8):
    for index, x in np.ndenumerate(tau):
        if x <= thres and x >= -thres:
            tau[index] = 0.0;
            
#        if x <= 0:
#            tau[index] = 0.0;
    return tau;
# return a list of the indices of nonzero entries of a matrix and their corresponding indices
def nonZeroEntries(mat):
    cleanMat = truncate(mat);
    nonZero = []
    for index, x in np.ndenumerate(cleanMat):
        if abs(cleanMat[index]) > 0.0:
            nonZero.append(index)
    return nonZero;

# generate constraints as a list of constraints
def constraints( indices, social, user):
    constraintList = [];
    for ind in range(len(indices)):
        stateActionPair = indices[ind];
        upperBound = social[stateActionPair] < user[stateActionPair] ;
        constraintList.append(Constraint(indices[ind], 
                                         social[stateActionPair], 
                                         upperBound));
                                         
    return constraintList;

# generate a matrix with all zeros and specific entries as dictated
# for each (t,s,a) is assigned uniquely
def matGen(constraintList, values, shapeList ):
    arr = np.zeros(shapeList);
    for ind in range(len(constraintList)):
        if constraintList[ind].upperBound:
            arr[constraintList[ind].index] = values[ind];
        else:
            arr[constraintList[ind].index] = -values[ind];
        
    return arr;
# generate a (state, action, time) matrix with tolls in the right spots
def toll2Mat(cState, tolls, shapeList, isLowerBound ):
    arr = np.zeros(shapeList);
    if isLowerBound == False:
        tolls = -1.*tolls;
    for ind in range(len(tolls)): # iterating over time
        arr[cState,:,ind] += tolls[ind];
        
    return arr;
# plot states's densities evolving in time
def statePlot(y):
    States, Actions, Time = y.shape;
    fig = plt.figure(); 
    timeLine = np.linspace(1,Time,20); 
    for s in range(States):
        traj = np.sum(y[s,:,:],axis=0)  
        plt.plot(timeLine,traj,label = r'state %d'%(s));
    
#    plt.title("Trajectory of State Densities with 60 people, state 6 constrained")
    plt.legend(fontsize = 'xx-small');
    plt.xlabel("Time");
    plt.ylabel("Optimal Driver Density")
    plt.show();

def cumulative_average(x_list):
    N = 1
    avg_list = [np.zeros(x_list[0].shape)]
    for x in x_list:
        next_avg = (avg_list[-1] * (N - 1) + x) / (N)
        avg_list.append(next_avg)
        N += 1
    avg_list.pop(0)
    return avg_list
        