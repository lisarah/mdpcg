# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:50:12 2019

@author: craba
"""
import gameSolvers.mystic as mys
import util.mdp as mdp

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
Time = 20;

sGame = mys.bilevel(Time);
seattleGraph=sGame.G;
p0 = np.zeros((seattleGraph.number_of_nodes()));
# make all drivers start from residential areas 6 of them
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

sGame.solve(p0);