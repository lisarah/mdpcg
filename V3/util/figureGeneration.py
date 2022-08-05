# -*- coding: utf-8 -*-
"""
Created on Sun Jun 03 13:12:30 2018

@author: craba
"""
# Import Required Packages
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def NeighbourGen(draw = True):
    # Number of nodes in graph:
    n = 12

    # Custom graph from edge list
    # Simple:
    # edges = [(1,2),(2,3),(3,4),(3,5),(2,5)]
    
    # With edge weights
    edges = [(1,2),  
             (1,4),
             (1,8),
             (1,11),
             (1,12),
             (2,6),
             (2,5),
             (2,7),
             (2,10),
             (3,4),
             (3,5),
             (3,11),
             (3,9),
             (4,5),
             (4,11),
             (5,7),
             (5,9),
             (6,7),
             (6,10),
             (8,12), 
             (11,12)]


    # Create graph object from edge list
    G = nx.Graph(edges)

    # User specified
    nodePos = {1:  (500,-275),
               2:  (450,-450),
               3:  (200,-175),
               4:  (350,-275),
               5:  (300,-400),
               6:  (400,-550),
               7:  (350,-500),
               8:  (600,-200),
               9:  (150,-350),
               10: (425,-600),
               11: (400,-175),
               12: (525,-200)}
    if draw:
        nx.draw(G, pos=nodePos, node_color='c', edge_color='k',
                font_weight='bold', transparent=True) # With specified positions
    distances = {};
    for e in edges:
        d1 = np.asarray(nodePos[e[0]]);
        d2 = np.asarray(nodePos[e[1]]);
        distances[e] = np.linalg.norm(d1 - d2);
#    # node labels
#    labels={}
#    labels[1]=r'UW'
#    labels[2]=r'Capitol Hill'
#    labels[3]=r'Ballard'
#    labels[4]=r'Fremont'
#    labels[5]=r'Queen Anne'
#    labels[6]=r'Pike Place'
#    labels[7]=r'Belltown'
#    labels[8]=r'Sand Point'
#    labels[9]=r'Magnolia'
#    labels[10]=r'Internation District'
#    labels[11]=r'Green Lake'
#    labels[12]=r'Ravenna'
#    nx.draw_networkx_labels(G,nodePos,labels,font_size=16)
    return nodePos, G, distances;

def airPortVis():
    """
        Visualize a state-action distribution over seattle airport
        see MDP.airportMDP for MDP definition
        
    """
    N = 12; # number of states
    # With edge weights
    edges = [(1,2),  
             (1,3),
             (2,3),
             (2,4),
             (4,5),
             (4,6),
             (5,6)];
    G = nx.Graph(edges)
    nodePos = {1:  (400, 200),
               2:  (200, 200),
               3:  (300,   0),
               4:  (100, 400),
               5:  (300, 500),
               6:  (200, 600)};
    nx.draw(G, pos=nodePos, node_color='b', edge_color='k',
                font_weight='bold', transparent=True) # With specified positions

    return G, nodePos;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    