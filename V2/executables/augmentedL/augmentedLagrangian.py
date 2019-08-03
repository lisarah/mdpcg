# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:23:20 2018

@author: craba
"""

import Algorithms.mdpRoutingGame as mrg
import Algorithms.augmentedLagrangianMethod as alm
import util.mdp as mdp
import util.utilities as ut
import Algorithms.frankWolfe as fw

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ------------- Part 0: Define parameters and solve social game ------------ #
sysStart = time.time();
Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);
socialGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=socialGame("G");
socialGame.setQuad();

p0 = 60.*mdp.resInit(seattleGraph.number_of_nodes());
socialRes, socialObj = socialGame.solve(p0, verbose=False,returnDual=False, isSocial = True );

#-----------Part 1: define feasible flows and constraints ------------------- #
userGame = mrg.mdpRoutingGame(seattle,Time);
userGame.setQuad();
R = userGame("reward"); 
C = userGame("C");

def gradF(x, constraints = None):
    # no constraint version
    return -np.multiply(R, x) + C;
    # constrained version

#------- Part 2: define the tolls required for social optimal equilibrium ----#
thresh = 1.;
f0 = np.zeros((socialGame.States,socialGame.Actions, Time));
y0, yHist = fw.FW(f0, p0, userGame("probability"), gradF,True); # y0 = state x action x time
       
tolls = 100.;    
diff = ut.truncate(socialRes- y0, tolls); # solution difference
nonZeroInd = ut.nonZeroEntries(diff ); # indices of non zero differences
constraintList = ut.constraints(nonZeroInd, socialRes, y0); # list of constraints
constraintNumber = len(constraintList);


#------------mu0---------------
mu0 = []; 
for ind in range(constraintNumber):
	sap = nonZeroInd[ind];
	diff = y0[sap] - socialRes[sap];
	isUpperbound = diff>0;
	mu0.append(ut.Constraint(sap,-diff, isUpperbound));

#------------c0---------------
c = 1.;
#c0 = np.sum(-0.5*np.multiply(np.multiply(R,y0),y0)+ np.multiply(C,y0));
#c0 = c0 /np.sum(np.multiply(nonZeroInd,nonZeroInd));
c0 = 10.
Cap = ut.truncate(socialRes); #  capacity constraints
# Initial mu guess = constraintList
# Initial flow guess = f0
# optCRes = sGame.solveWithConstraint(p0,verbose = False, constraintList = constraintList);
# optDual = ut.matGen(constraintList, sGame.optimalDual, [sGame.States, sGame.Actions, Time]);
# tolledObj[incre] = sGame.socialCost(optCRes); # +  tolls[incre]* np.sum(np.multiply(optDual, optCRes));
# print "Social cost at constrained User Optimal";
# print tolledObj[incre];
muStar, error, fStar = alm.ALM(c0, mu0, f0, p0, R, C, userGame("probability"), Cap)


print np.sum(-np.multiply(R, np.multiply(fStar, fStar)) + np.multiply(fStar, C))

plt.figure();
plt.plot(error/3600.);

# ------------------ use the optimal dual parameters to MSA------------------#
nonZeroDiff = diff;
optCRes = userGame.solveWithConstraint(p0,verbose = False, constraintList = constraintList);
