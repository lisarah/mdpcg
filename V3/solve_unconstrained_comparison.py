# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:50:12 2019

@author: craba
"""
import gameSolvers.mystic as mys
import gameSolvers.cvx as cvx
import util.mdp as mdp

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
Time = 20;

sGame = mys.mysticGame(Time);
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

solution = sGame.solve(p0);
solution = np.reshape(solution, (sGame.states, sGame.actions, sGame.Time));

# compare to the cvx solution
cvxGame = cvx.cvxGame(Time);
cvxGame.R = 1.0*sGame.R;
cvxGame.C = 1.0*sGame.C;
cvxGame.P = 1.0*sGame.P;

p0 = np.zeros((seattleGraph.number_of_nodes()));
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

print ("Solving primal unconstrained case");
optRes, mdpRes = cvxGame.solve(p0, verbose=False,returnDual=False);