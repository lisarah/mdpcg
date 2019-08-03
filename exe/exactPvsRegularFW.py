# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:06:00 2018

@author: craba
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:10:25 2018

@author: sarah
"""
import Algorithms.mdpRoutingGame as mrg
import util.mdp as mdp
import Algorithms.dynamicProgramming as dp
import util.utilities as ut
import Algorithms.frankWolfe as fw

import numpy as np
import numpy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt

Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);

sGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=sGame("G");
sGame.setQuad();
numPlayers = 100;
p0 = mdp.resInit(seattleGraph.number_of_nodes(), residentialNum=6./numPlayers);
print "Solving primal unconstrained case";
optRes, optObj = sGame.solve(p0, verbose=False,returnDual=False);


cState = 6;   cThresh = 10;                             
sGame.setConstrainedState(cState, cThresh, isLB = True);
print "Solving constrained case, state 7 >= 10 case";
optCRes = sGame.solveWithConstraint(p0,verbose = False);
print "optimal dual: ", sGame("optDual")
print "lower bound" , sGame("lowerBound")

cleanToll = abs(ut.truncate(sGame("optDual")));
cleanToll = np.concatenate((np.array([0,0,0]), cleanToll));
barC = ut.toll2Mat(cState, cleanToll, [sGame.States, sGame.Actions, sGame.Time], True);

# Simulate the values converging to optimal solution
threshIter = 1;
ytThresh = np.zeros((sGame.States, sGame.Actions, sGame.Time, threshIter));
threshVal = np.zeros((threshIter));
normDiff = np.zeros((threshIter, 2)); # two norm and infinity norm
ytHist = None;

# FW of values converging
threshVal = 1e-3;
def gradF(x):
  return -np.multiply(sGame("reward"), x) + sGame("C") + barC;


#-------------------- With regular Penalty -------------------#
x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
ytThresh, ytHist = fw.FW(x0, p0, sGame("probability"), gradF, True, threshVal);
ytHistArr = np.zeros(len(ytHist));
for i in range(len(ytHist)):
    ytHistArr[i] = la.norm((ytHist[i] - optCRes));
#fig = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';
#plt.plot(np.linspace(1, len(ytHist),len(ytHist)), ytHistArr, linewidth = 2, label = r'$||\cdot||_2$',color = blue);
#plt.legend();
##plt.title("Difference in Norm as a function of termination tolerance")
#plt.xlabel(r"Iterations");
#plt.ylabel(r"$\frac{||y^{\epsilon} - y^{\star}||}{||y^{\star}||}$");
#plt.xscale('log')
##plt.xscale("log");
#plt.grid();
#plt.show();

#-------------------- With state constraints -------------------------------#
fig = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';
plt.plot(np.linspace(1, len(ytHist),len(ytHist)), ytHistArr/la.norm(optCRes), linewidth = 2, label = r'regular penalty',color = blue);

for j in range(5):
    curDelta = 0.1**j + 1;
    def exactGrad(x):
        grad = -np.multiply(sGame("reward"), x)+ sGame("C");
        for time in range(Time):
            xDensity = np.sum(x[cState,:,time]);
            if xDensity<= cThresh: # put actual constraint here
                grad[cState,:,time] += curDelta*barC[cState,:,time]
        return grad;
    # This is the not state constrained case
    x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
    ytCThresh, ytCHist = fw.FW(x0, p0, sGame("probability"), exactGrad, True, threshVal);
    ytCHistArr = np.zeros(len(ytCHist));
    for i in range(len(ytCHist)):
        ytCHistArr[i] = la.norm((ytCHist[i]  - optCRes));
    plt.plot(np.linspace(1, len(ytHist),len(ytCHist)), ytCHistArr/la.norm(optCRes), linewidth = 2, label = str(curDelta - 1));

plt.legend();
#plt.title("Difference in Norm as a function of termination tolerance")
plt.xlabel(r"Iterations");
plt.ylabel(r"$\frac{||y^{\epsilon} - y^{\star}||}{||y^{\star}||}$");
#plt.xscale('log')
plt.yscale("log");
plt.grid();
plt.show();
