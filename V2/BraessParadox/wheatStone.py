# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:03:43 2018

@author: craba
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import supFunc as sF
# Graph set up 
edges = [(0,1),(0,2), (1,2), (2,3), (1,3)] ;
edges2 = [(0,1),(0,2), (2,3), (1,3)] ; # removed middle edge
wheat = nx.DiGraph(edges);
wheatRemoved = nx.DiGraph(edges2);
# Degree/Incidence matrix
E = nx.incidence_matrix(wheat, oriented=True).toarray();
E2 = nx.incidence_matrix(wheatRemoved, oriented=True).toarray();
# Number of edges
e = wheat.number_of_edges();
# Number of nodes
v = wheat.number_of_nodes();
#Specify their positions
nodePos = {0: (0,1),
           1: (1,2),
           2: (1,0),
           3: (2,1)}
# Draw graph
#nx.draw(wheat,pos=nodePos, node_color='g', edge_color='k', with_labels=True, font_weight='bold', font_color ='w')
#plt.show();
sourceNode = 0;
sinkNode = 3;
sourceVec = np.zeros((v));
sourceVec[sourceNode] = -1;
sourceVec[sinkNode] = 1;

#-------------latency parameters----------------------#
eps=0.001;
A = np.diag([1., eps, eps, 1., eps, eps]) #latency for whole graph
b = np.array([eps, 1.0,1.1, eps, 1.]) 

#A2 = np.diag([5.0, 1.0, 0.1, 5.0, 5.0]) #latency for whole graph
#b2 = np.array([0., 5.0, 1., 0., 5.0]) 

#sF.RouteGen(wheat,edges,sourceNode,sinkNode)
#
#socialCosts = sF.returnPotValues(50, 0.1, sF.Q, wheat, edges, sourceNode, sinkNode, A,b);  
#warCosts = sF.returnPotValues(50, 0.1, sF.P, wheat, edges, sourceNode, sinkNode, A,b);    
#plt.figure()
#plt.plot(socialCosts[0,:],socialCosts[1,:],label=("Social Potential"));
#plt.plot(warCosts[0,:],warCosts[1,:],label=("Wardrop Potential"));
#plt.xlabel('Mass Input')
#plt.ylabel('Potential Value')
#plt.title('Potential Comparison')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show();

# Braess Paradox
totalIterations = 50; increment = 0.001;
socialFull = sF.iterateStochasticB(totalIterations, increment, sF.Q, wheat, edges, sourceNode, sinkNode, A,b);  
#socialNoEdge = sF.iterateB(totalIterations, increment, sF.Q, wheatRemoved, edges2, sourceNode, sinkNode, ARemoved,bRemoved); 

#plt.figure()
#plt.plot(socialFull[7,:], socialFull[6,:] ,label=("Social Potential At Wardrop"));
#plt.plot(socialFull[7,:], socialFull[0,:], label = ("edge 1"));
#plt.plot(socialFull[7,:], socialFull[1,:], label = ("edge 2"));
#plt.plot(socialFull[7,:], socialFull[2,:], label = ("edge 3"));
#plt.plot(socialFull[7,:], socialFull[3,:], label = ("edge 4"));
#plt.plot(socialFull[7,:], socialFull[4,:], label = ("edge 5"));
#plt.plot(socialFull[7,:], socialFull[5,:], label = ("total mass"));
##plt.plot(socialFull[0,:],socialNoEdge[1,:],label=("Social Potential After removing the edge"));
#
#plt.xlabel('BValue')
#plt.ylabel('Social Value')
#plt.title('Potential Comparison:')
#plt.legend();
#plt.grid();
##plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show();

F = -np.array([[ 1.,  0.,  0.,   0.,  1., -1.], 
              [ -1.,  1.,  1.,   0.,  0.,  0.], 
              [  0.,  0.,  -0.6,   1., -1.,  0.], 
              [  0., -1.,   -0.4,   -1., 0.,  1.]]);
#N = np.array([[ 1.,  0.,  0.,   0.,  1.], 
#              [ -1.,  1.,  1.,   0.,  0.], 
#              [  0.,  0.,  -0.6, 1., -1.], 
#              [  0., -1.,  -0.4, -1., 0.],]);
N = np.array([[ 1.,  0.,  0.,   0.,  1., -1.], 
              [ -1.,  1.,  1.,   0.,  0.,  0.], 
#              [  0.,  0.,  -0.6, 1., -1.,  0.], 
              [  0., -1.,  -0.4, -1., 0.,  1.],
              [  1.,  1.,  1.,   1.,  1.,  1.]]);
pA = np.diag([1.0, 0., 0., 1.0, 0., 0.]);
GInv = np.linalg.inv(A);
sumFlow = socialFull[0,:] + socialFull[1,:] +socialFull[2,:] + socialFull[3,:] + socialFull[4,:] + socialFull[5,:]; # + 
NGN = np.linalg.inv(N.dot(GInv).dot(N.T))
yStar = socialFull[0:6, :];

bMat = np.tile(np.concatenate((b, np.array([0]))), (totalIterations,1));
bMat[:,2] = socialFull[7,:];
lStar = A.dot(yStar).T + bMat;# np.concatenate((b, np.array([0])));

sensitivity = GInv.dot((N.T.dot(NGN.dot(N.dot(yStar))))) + GInv.dot(N.T.dot(NGN.dot(N.dot(GInv.dot(lStar.T))))) - GInv.dot(lStar.T)

plt.figure()
plt.plot(socialFull[7,:], socialFull[6,:] ,label=("Social Potential At Wardrop"));
plt.plot(socialFull[7,:], socialFull[0,:], label = ("edge 1"));
plt.plot(socialFull[7,:], socialFull[1,:], label = ("edge 2"));
plt.plot(socialFull[7,:], socialFull[2,:], label = ("edge 3"));
plt.plot(socialFull[7,:], socialFull[3,:], label = ("edge 4"));
plt.plot(socialFull[7,:], socialFull[4,:], label = ("edge 5"));
plt.plot(socialFull[7,:], socialFull[5,:], label = ("edge 6"));
#plt.plot(socialFull[7,:], sumFlow, label = ("total flow on network"));
plt.plot(socialFull[7,:], sensitivity[2,:], label = ("Sensitivity"));
#plt.plot(socialFull[0,:],socialNoEdge[1,:],label=("Social Potential After removing the edge"));

plt.xlabel('BValue')
plt.ylabel('Social Value')
plt.title('Potential Comparison:')
plt.legend();
plt.grid();
plt.show();