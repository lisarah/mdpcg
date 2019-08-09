# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:42:34 2019

@author: craba
"""

import gameSolvers.mystic as mys
import numpy as np


import gc
gc.collect();
Time = 20;

sGame = mys.bilevel(Time);
seattleGraph = sGame.G;
p0 = np.zeros((seattleGraph.number_of_nodes()))
residentialNum= 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

yOpt, epsOpt = sGame.solve(p0);

np.save("optimalY", yOpt, allow_pickle=False);
np.save("optimalEps", epsOpt, allow_pickle = False);

# ---------- to load file again---------------#
#yOpt = np.load("optimalY.npy");
#optimalEps = np.load("optimalEps.npy");

