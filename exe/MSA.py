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
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optRes,
#                          startAtOne = True,
#                          numPlayers= p0[0]);
cState = 6;      
cThresh = 0.2*numPlayers;   
isLB = True;                      
sGame.setConstrainedState(cState, cThresh, isLB = isLB);
print "Solving constrained case, state 7 >= 150 case";
optCRes = sGame.solveWithConstraint(p0,verbose = False);
#print "optimal dual: ", sGame("optDual")
#print "upper bound" , sGame("constrainedUpperBound")
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCRes/10.,
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("constrainedUpperBound"));
if isLB: 
    optimalDual = np.concatenate((np.zeros(3), sGame("optDual") )) + 0.01;
else:
    optimalDual = -np.concatenate((np.zeros(3), sGame("optDual"))) + 0.01;
#####
print "Solving unconstrained problem with new Toll";
optCSol, optTolledObj = sGame.solve(p0,withPenalty=True,verbose = False, returnDual = False)
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCSol/10., 
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
##                          constrainedUpperBound = sGame("constrainedUpperBound"));
##     

cleanOpt = abs(ut.truncate(optCSol));
barC =  sGame("C") + ut.toll2Mat(cState, optimalDual, [sGame.States, sGame.Actions, sGame.Time], isLB);

# Simulate the values converging to optimal solution
threshIter = 1;
ytThresh = np.zeros((sGame.States, sGame.Actions, sGame.Time, threshIter));
threshVal = np.zeros((threshIter));
normDiff = np.zeros((threshIter, 2)); # two norm and infinity norm
ytHist = None;
def gradF(x):
  return -np.multiply(sGame("reward"), x) + barC;



for iter in range(threshIter):
    print "iteration = ", iter;
#    threshVal[iter] = 91.0 - (iter/2.)**2; 
    threshVal[iter] = 1e-3;
    x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
    ytThresh[:,:,:,iter], ytHist = fw.FW(x0, p0, sGame("probability"), gradF, threshVal[iter], maxIterations = 2000);
    yDP = abs(ut.truncate(ytThresh[:,:,:,iter]));
    normDiff[iter, 0] = la.norm((ytThresh[:,:,:,iter] - optCSol));
    normDiff[iter, 1] = (abs(ytThresh[:,:,:,iter] - optCSol)).max();      
timeLine = np.linspace(1,Time,20);
threshNormDiff = np.zeros((threshIter,2));

start = 3;
#plt.plot(timeLine[start:],np.sum(optCSol[cState,:,start:],axis=0),label = "primal constrained trajectory"); 
for i in range(threshIter):
#    print i;
    diff = (ytThresh[cState,:,:, i])- (optCSol[cState,:,:]);
    threshNormDiff[i,0] = la.norm(diff);
    threshNormDiff[i,1] = np.max(abs(diff));
#    if i%4 == 0:         
#        plt.plot(timeLine[start:],np.sum(ytThresh[cState,:,start:, i],axis=0),dashes=[4, 2],label =r'$\epsilon$ = %.2f'%(threshVal[i])); 
    
##plt.legend(fontsize = 'xx-small');
cStarNorm = np.linalg.norm(gradF(optCRes));
#fig = plt.figure();  
#plt.plot(threshVal/cStarNorm, threshNormDiff/(np.linalg.norm(optCSol[cState,:,:])), linewidth = 2);
##plt.title("Optimal Dual Solution with Decreasing Termination Tolerance")
#plt.xlabel(r"$\epsilon$");
#plt.ylabel(r"$\frac{||y^{\epsilon} - y^{\star}||_2}{||y^{\star}||_2}$");
##plt.yscale("log");
##plt.xscale("log");
#plt.grid();
#plt.show();

fig = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';
plt.plot(threshVal/cStarNorm, normDiff[:,0]/(la.norm(optCSol)), linewidth = 2, label = r'$||\cdot||_2$',color = blue);
#plt.plot(threshVal/cStarNorm, normDiff[:,1]/(np.max(abs(optCSol))), linewidth = 2, label = r'$||\cdot||_{\infty}$', color = orange);
plt.plot(threshVal/cStarNorm, threshNormDiff[:,0]/(la.norm(optCSol[cState,:,:])), label =r'$||\cdot||_2 $, state 7', linewidth = 2, linestyle = ':', color = blue);
#plt.plot(threshVal/cStarNorm, threshNormDiff[:,1]/np.max(abs(optCSol[cState,:,:])), label =r'$||\cdot||_{\infty}$, state 7', linewidth = 2, linestyle = ':', color = orange);
plt.legend();
#plt.title("Difference in Norm as a function of termination tolerance")
plt.xlabel(r"$\epsilon$");
plt.ylabel(r"$\frac{||y^{\epsilon} - y^{\star}||}{||y^{\star}||}$");
#plt.xscale("log");
plt.grid();
plt.show();

fig = plt.figure();
ytHistArr = np.zeros((len(ytHist), 2));
for i in range(len(ytHist)):
    ytHistArr[i, 0] = la.norm((ytHist[i] - optCSol));
    ytHistArr[i, 1] = la.norm((ytHist[i][cState,:,:])- (optCSol[cState,:,:]));
    
plt.plot(np.linspace(1, len(ytHist),len(ytHist)), ytHistArr[:, 0]/(la.norm(optCSol)), linewidth = 2, label = r'$||\cdot||_2$',color = blue);
#plt.plot(np.linspace(1, len(ytHist),len(ytHist)), ytHistArr[:, 1]/(la.norm(optCSol[cState,:,:])), label =r'$||\cdot||_2 $, state 7', linewidth = 2, linestyle = ':', color = blue);
#plt.legend();
#plt.title("Difference in Norm as a function of termination tolerance")
plt.xlabel(r"Iterations");
plt.ylabel(r"$||y^{\epsilon} - y^{\star}||_2$");
plt.yscale('log')
#plt.xscale('log');
plt.grid();
plt.show();