#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:36:33 2019

@author: sarahli
"""
import numpy as np
import networkx as nx
import Algorithms.frankWolfe as fw
def dec_admm(p0,game, graph, N, penalty = 1.0):
    maxDeg = len(nx.degree_histogram(graph))
    y = np.zeros([graph.number_of_nodes(), maxDeg]);
    tau = np.zeros([graph.number_of_nodes(), graph.number_of_nodes(), maxDeg]);
    # total iteration
    for k in range(N):
        # primal variable update
        for state in range(1,graph.number_of_nodes()+1):
            yk = y[state-1,:];
            # form cost
            A = game.R[state-1,:]; B = game.C[state-1,:];
            neighbours = list(graph.neighbors(state));
            for neighbor in neighbours:
                i = max(state,neighbor);
                j = min(state,neighbor);
                A += penalty;
                B += tau[i-1,j-1,:] - penalty*y[j-1,:];
                
            def gradF(y):
                return A*y + B;
            
            print "Solve with Frank Wolfe for state ", state;
            yk, xHist = fw.localFW(0, p0[state-1], game.P[:,state-1,:], gradF, maxIterations = 100);
            y[state-1,:] = yk;
        # dual variable update    
        for state in range(1,graph.number_of_nodes() +1):
            for neighbour in list(graph.neighbors(state)):
                i = max(state,neighbour);
                j = min(state,neighbour);
                yi = y[i-1,:]; yj = y[j-1,:];
                tauk = 1.0*tau[i-1,j-1,:];
                tau[i-1,j-1,:] = tauk + penalty*(yi - yj);
                
    return y;