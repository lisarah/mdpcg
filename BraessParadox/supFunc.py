# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:21:08 2018

@author: craba
"""
import cvxpy as cvx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def RouteGen(graph, edgeSet, sourceN, sinkN):
    routes =  list(nx.all_simple_paths(graph, source=sourceN, target=sinkN));
    RouteMat = np.zeros((graph.number_of_edges(), len(routes)));
    for route in range(0,len(routes)):
        curRoute = routes[route];#Route is in nodes
        #Look for the index of corresponding edge
        for edge in range(0,len(curRoute)-1):
            #Unpack edge
            start= curRoute[edge]; end = curRoute[edge+1];
            edgeInd = edgeSet.index((start,end)); #find the edge index in list
            RouteMat[edgeInd,route]= 1;
    return RouteMat;

def latency(x, A, b): 
    return A.dot(x) +b;
def P(x, A, b): # Wardrop Potential
    return 0.5*x.T.dot(A).dot(x) + b.T.dot(x);
def Q(x, A, b):  # Average Potential
    quadTerm = x.T.dot(A.dot(x)); 
    
    return quadTerm + b.T.dot(x);

# return a function value at the wardrop equilibrium defined using l(x) = Ax + b
def returnPotValues(massIterations, incre, potFunc,graph, edgeSet, sink, source, A,b):
    # returns an array of potential values at wardrop Equilibrium 
    potVals = np.zeros([2,massIterations]);
    RouteMat = RouteGen(graph, edgeSet, sink,source);
    Incidence = nx.incidence_matrix(graph, oriented = True);
    e,r = RouteMat.shape;
    for massInd in range(0,massIterations):
        # Problem data.
        mass = incre*massInd+0;
        potVals[0,massInd] = mass;
        # Construct the problem.
        #----------------WARDROP EQUILIBRIUM--------------------
        x = cvx.Variable(e); 
        z = cvx.Variable(r);
        warPot =  b*x + cvx.quad_form(x,A)*0.5;
        warObj = cvx.Minimize(warPot);
        warConstraints = [0 <= z, 
                          RouteMat*z == x,
                          sum(z) == mass]
        wardrop = cvx.Problem(warObj, warConstraints)
        warRes = wardrop.solve(solver=cvx.MOSEK)
        # Return the potential value for desired potential
        potVals[1,massInd] = potFunc(x.value,A,b);
        
    return potVals;

# return a function value at the wardrop equilibrium defined using l(x) = Ax + b
def iterateB(bIterations, incre, potFunc, graph, edgeSet, sink, source, A,b):
    # returns an array of potential values at wardrop Equilibrium 
    potVals = np.zeros([4,bIterations]);
    RouteMat = RouteGen(graph, edgeSet, sink,source);
    e,r = RouteMat.shape;
    mass = 0.25;
    for bInd in range(0,bIterations):
        # Problem data.
        bVal = incre*bInd*np.array([0,0,1,0,0]) + b;
        potVals[0,bInd] = incre*bInd + b[2];
        # Construct the problem.
        #----------------WARDROP EQUILIBRIUM--------------------
        x = cvx.Variable(e); 
        z = cvx.Variable(r);
        warPot =  bVal*x + cvx.quad_form(x,A)*0.5;
        warObj = cvx.Minimize(warPot);
        warConstraints = [0 <= z, 
                          RouteMat*z == x,
                          z[5] == mass]
        wardrop = cvx.Problem(warObj, warConstraints)
        warRes = wardrop.solve(solver=cvx.MOSEK)
        # Return the potential value for desired potential
        potVals[1,bInd] = potFunc(x.value,A,bVal);
        potVals[2,bInd] = x.value[2];
        potVals[3,bInd] = x.value[0];
    return potVals;

def iterateStochasticB(bIterations, incre, potFunc, edgeSet, sink, source, A,b):
    # returns an array of potential values at wardrop Equilibrium 
    potVals = np.zeros([8,bIterations]);
#    RouteMat = RouteGen(graph, edgeSet, sink,source);
#    e,r = RouteMat.shape;
    incidence = np.array([[ 1.,  0.,  0.,   0.,  1., -1.], 
                          [ -1.,  1.,  1.,   0.,  0.,  0.], 
                          [  0.,  0.,  -0.9,   1., -1.,  0.], 
                          [  0., -1.,   -0.1,   -1., 0.,  1.]]);
#    i2 = np.array([[ 1.,  0.,  0.,   0.,  1.], 
#                  [ -1.,  1.,  1.,   0.,  0.], 
#                  [  0.,  0.,  -0.6,   1., -1.], 
#                  [  0., -1.,   -0.4,   -1., 0.]]);
    e,r = incidence.shape;
    mass = 1.0;
    for bInd in range(0,bIterations):
        # Problem data.
        bVal = incre*bInd*np.array([0,1.,0,0,0]) + b;
        print bVal;
        potVals[0,bInd] = incre*bInd + b[2];
        bVal = np.concatenate((bVal, np.array([0])), axis = None);
#        print bVal.shape;
#        print r;
        # Construct the problem.
        #----------------WARDROP EQUILIBRIUM--------------------
#        x = cvx.Variable(e); 
        z = cvx.Variable(r);
        warPot =  sum([bVal[i]*z[i] for i in range(r)])+ cvx.quad_form(z,A)*0.5;
        warObj = cvx.Minimize(warPot);
        warConstraints = [0. <= z, 
                          sum([incidence[0, i]*z[i] for i in range(r)]) == 0.,
                          sum([incidence[1, i]*z[i] for i in range(r)]) == 0.,
                          sum([incidence[2, i]*z[i] for i in range(r)]) == 0.,
#                          sum([incidence[3, i]*z[i] for i in range(r)]) == 0.,#];
                          sum([z[i] for i in range(r)]) == mass];
#                          z[5] == mass]
        wardrop = cvx.Problem(warObj, warConstraints)
        warRes = wardrop.solve(verbose = False)
        # Return the potential value for desired potential
        potVals[0,bInd] = z.value[0];
        potVals[1,bInd] = z.value[1];
        potVals[2,bInd] = z.value[2];
        potVals[3,bInd] = z.value[3];
        potVals[4,bInd] = z.value[4];
        potVals[5,bInd] = z.value[5];
        potVals[6,bInd] = potFunc(z.value,A,bVal);
        potVals[7,bInd] = incre*bInd + b[2];
    return potVals;