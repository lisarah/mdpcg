# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:39:08 2018

@author: craba
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 12:09:47 2018

@author: sarah
These are helpers for mdpRouting game class
"""
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import networkx as nx

def erdosRenyiMDP(nodes, edges):
    G = nx.gnm_random_graph(nodes, edges);
    degree_sequence=list(nx.degree(G))
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    # print "Degree sequence", degree_sequence
    dmax = max(degree_sequence);


    A = np.asarray(nx.adjacency_matrix(G).todense());
    P = np.zeros((nodes, nodes,dmax));
    p = 0.8;
    # ------------ probability generation --------------- #
    for node in range(nodes):
        neighbours = list(G.neighbors(node));
        totalN = len(neighbours);
        pNot = (1.-p)/(totalN);
        actionIter = 0;
        for neighbour in neighbours: 
            P[neighbour,node,actionIter] = p;
            for scattered in neighbours:
                if scattered != neighbour:
                    # probablity of ending up at a neighbour
                    P[scattered,node,actionIter] = pNot;  
            P[node,node,actionIter] =pNot;
            actionIter +=1;
        while actionIter <dmax:
            P[:,:,actionIter] = P[:,:,actionIter-1];
            actionIter+=1;
#    # ------------ probability generation --------------- #
#    for action in range(dmax):
#        chance = np.random.rand(nodes, nodes);
#        P[:,:,action] = np.multiply(A, chance);
#        for fromState in range(nodes):
#            P[:,fromState, action] = 1. / np.sum(P[:,fromState,action])*P[:,fromState, action];
            
    # ------------- phi generation ---------------------- #
    # phi = S multiply y + Sc
    # S is states x actions
    S = 5.*np.random.rand(nodes, dmax) + 1.0;
    Sc = 10.* np.random.rand(nodes, dmax) ;

    # ------------- psi generation ---------------------- #
    # psi_inv = exp(multiply(D,y))
    # S is states x actions
    D = -4.0* np.random.rand(nodes)-  1.0;
    Dc = 400.*np.ones(nodes);
    return G, P, S, Sc, D,Dc;
    
    
    