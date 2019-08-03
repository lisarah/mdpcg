#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:36:33 2019

@author: sarahli
"""
import Algorithms.infMDP as imdp
#import Algorithms.dynamicProgramming as dp
import Algorithms.admm as admm
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt

##------------------- Set up MDP Routing game  ----------------------------#
seattle = imdp.gParam("seattleQuad", None, None);
sGame = imdp.infMDP(seattle, beta = 1.0);
seattleGraph=sGame.G;
#nx.draw(seattleGraph, pos = sGame("graphPos"),with_labels=True);
#plt.show()
#p0 = np.ones((seattleGraph.number_of_nodes()))/seattleGraph.number_of_nodes();
p0 = np.zeros((seattleGraph.number_of_nodes()));
#p0[0] = 1.0;
# make all drivers start from residential areas 6 of them
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

discount  = 0.7;
sGame = imdp.infMDP(seattle, beta =  discount);
infObj, infRes = sGame.solve(p0,verbose=False,returnDual=False);
    
##------------------- solve with decentralized algorithm  ----------------------------#
yApprox = admm.dec_admm(p0, sGame, seattleGraph, 200, penalty = 0.5)
##------------------- Plot 2: Total value of income ----------------------------
fig = plt.figure();  
plt.bar(np.linspace(1,seattleGraph.number_of_nodes(),seattleGraph.number_of_nodes()),np.sum(yApprox,axis=1));                      
plt.title("Steady State distribution")
plt.xlabel("States");
plt.ylabel("Density")
plt.show();
