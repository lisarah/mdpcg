# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 12:56:58 2018

@author: craba
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import Algorithms.mdpRoutingGame as mrg
import Algorithms.frankWolfe as fw
import util.mdp as mdp
import Algorithms.dualAscent as da
import util.utilities as ut
import math as math
plt.close('all');
Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);
#----------------set up game -------------------#
sGame = mrg.mdpRoutingGame(seattle,Time,strictlyConvex=False);
seattleGraph=sGame("G");
sGame.setQuad();
numPlayers = 100;
p0 = mdp.resInit(seattleGraph.number_of_nodes(), residentialNum=6./numPlayers);
##---------------set up NOT constrained game -----------------#
print "Solving primal unconstrained case";
optRes, mdpRes = sGame.solve(p0, verbose=False,returnDual=False);

#---------------set up constrained game -----------------#
cState = 6;   cThresh = 10;                             
sGame.setConstrainedState(cState, cThresh, isLB = True);
optCRes = sGame.solveWithConstraint(p0,verbose = False);
cleanToll = abs(ut.truncate(sGame("optDual")));
cleanToll = np.concatenate((np.array([0,0,0]), cleanToll));
barC = ut.toll2Mat(cState, cleanToll, [sGame.States, sGame.Actions, sGame.Time], True);



#---------------set up admm algorithm-------------------#
testN = 2;
dualPlot = plt.figure(1); plt.grid();
densityPlot = plt.figure(2); plt.grid();
certificatePlot = plt.figure(3);
constraintViolationPlot = plt.figure(4);

rhoVal = np.linspace(1, 2, testN);
maxIter  = 500;
Iterations= np.linspace(1,maxIter, maxIter);
timeLine = np.linspace(1,Time,Time);
lamb1 = None;
def gameObj(yk, rho, length = 1):
    obj = None;
    if length  == 1:
        obj = np.sum(0.5*np.multiply(np.multiply(sGame.R, sGame.R),yk) + np.multiply(sGame.C,yk)); # + rho/2*la.norm(np.sum(yk[cState,:,3:Time],axis=0) - 10*np.ones(Time-3))**2
    else:
        obj = np.zeros(length);
        for i in range(length):
            yki = yk[:,:,:,i];
            obj[i] =  np.sum(0.5*np.multiply(np.multiply(sGame.R, sGame.R),yki) + np.multiply(sGame.C,yki)); #  + rho/2*la.norm(np.sum(yki[cState,:,3:Time],axis=0) - 10*np.ones(Time-3))**2
    return obj;

def constraintResidual(yk, length):
    res = np.zeros(length);
    for i in range(length):
        yki = yk[:,:,:,i];
        for j in range(Time):
            if j >= 3:
                res[i] += np.square(np.sum(yki[cState,:, j]) - cThresh)
    return res;
        
for penaltyIter in range(testN):
    rho = rhoVal[penaltyIter];
    print "solving penalty ", rho;
    lambda0 = np.zeros((sGame.States, sGame.Actions,sGame.Time));
    y0 = np.zeros((sGame.States, sGame.Actions, sGame.Time)); 
     
    for i in range(3,Time):
        lambda0[cState,:, i] += 600.;
    
    yHist, lambHist, certificate,finalLamb = da.admm(lambda0, rho, y0, p0, sGame.P, cState,cThresh, sGame.R, sGame.C, maxErr = 1.0, optVar= barC,maxIterations = maxIter);
    a,b,c,d = yHist.shape;
    
   
    constrainedCost = gameObj(yHist,1, d);
    averageConstrainedCost = 1.0*constrainedCost;
    for i in range(len(constrainedCost)) :
        averageConstrainedCost[i] = np.sum(constrainedCost[0:i])/(i+1);
    constraintViolation = constraintResidual(yHist, d);
    

    plt.figure(2); # cost convergence
    plt.plot(Iterations,abs(gameObj(optCRes, rho) - averageConstrainedCost)/gameObj(optCRes,rho) , label = '%.1f'%rho );
    plt.yscale('log');
    plt.ylabel(r"$\frac{|L^\star - L^k|}{L^\star}$");
    plt.xlabel("Iterations");

    
    plt.figure(1);# dual convergence
    plt.plot(Iterations,1.0*lambHist/la.norm(barC), label = '%.2f'%rho);
    plt.yscale('log');
    plt.ylabel(r"$\frac{\|\tau^k - \tau^\star\|}{\|\tau^\star\|}$");
    plt.xlabel("Iterations");
    
    plt.figure(3);
    plt.plot(Iterations, 1.0*certificate/gameObj(optCRes,rho), label = '%.2f'%rho);
    plt.grid();
    
    plt.figure(4)
    plt.plot(Iterations, constraintViolation, label = '%.2f'%rho);
    plt.yscale('log');
    plt.ylabel(r"$\|Ay - b\|^2$");
    plt.xlabel("Iterations");

    if penaltyIter == 0:
        lamb1 = finalLamb;
    
#    plt.figure(4);# constrained state mass
#    axs[0].plot(timeLine, np.sum(ytThresh[cState,:,:],axis=0), linewidth = 2, linestyle = "-.", label ='exact penalty FW');

    
#plt.figure(3);
#plt.plot(Iterations, 1.0*certificate/gameObj(optCRes,rho), label = '%.2f'%rho);
#plt.grid();

densityPlot.legend();
densityPlot.show();

dualPlot.legend();
dualPlot.show();

certificatePlot.legend();
certificatePlot.show();



#------------------ apply regular penalty -----------------------#
# FW of values converging
threshVal = 1e-3;
#def gradF(x):
#  return -np.multiply(sGame("reward"), x) + sGame("C") + finalLamb;
#x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
#ytThresh, ytHist = fw.FW(x0, p0, sGame("probability"), gradF, True, threshVal, maxIterations = 500);
#ytHistArr = np.zeros(len(ytHist));
#for i in range(len(ytHist)):
#    ytHistArr[i] = la.norm((ytHist[i] - optCRes));
#    
#fig = plt.figure();
#blue = '#1f77b4ff';
#orange = '#ff7f0eff';

#curDelta = 1.1;
#def exactGrad(x):
#    grad = -np.multiply(sGame("reward"), x)+ sGame("C");
#    for time in range(Time):
#        xDensity = np.sum(x[cState,:,time]);
#        if xDensity< cThresh: # put actual constraint here
#            grad[cState,:,time] += curDelta*finalLamb[cState,:,time]
#    return grad;
## This is the not state constrained case
#x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
#ytCThresh, ytCHist = fw.FW(x0, p0, sGame("probability"), exactGrad, True, threshVal, maxIterations = 1000);
#ytCHistArr = np.zeros(len(ytCHist));
#for i in range(len(ytCHist)):
#    ytCHistArr[i] = la.norm((ytCHist[i]  - optCRes));
#plt.plot(np.linspace(1, len(ytHist),len(ytCHist)), ytCHistArr/la.norm(optCRes), linewidth = 2, label = str(curDelta - 1));
    
#---------------- exact penalty -------------------------#
plt.figure();
finalytCThresh = None;
for j in range(5):
    curDelta = 0.1**j + 1;
    def exactGrad(x):
        grad = -np.multiply(sGame("reward"), x)+ sGame("C");
        for time in range(3,Time):
            xDensity = np.sum(x[cState,:,time]);
            if xDensity< cThresh: # put actual constraint here
                grad[cState,:,time] += curDelta*lamb1[cState,:,time]
        return grad;
    # This is the not state constrained case
    x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
    ytCThresh, ytCHist = fw.FW(x0, p0, sGame("probability"), exactGrad, True, threshVal, maxIterations = 500);
    ytCHistArr = np.zeros(len(ytCHist));
    for i in range(len(ytCHist)):
#        ytCHistArr[i] = la.norm((ytCHist[i]  - optCRes));
        ytCHistArr[i] = gameObj(ytCHist[i], 1);
    averagedCYt = 1.0*ytCHistArr;
    for i in range(len(ytCHistArr)) :
        averagedCYt[i] = np.sum(ytCHistArr[0:i])/(i+1);
    plt.plot(np.linspace(1, len(ytCHist),len(ytCHist)), abs(gameObj(optCRes, 1) - averagedCYt)/gameObj(optCRes,1), linewidth = 2, label = str(curDelta - 1));
    finalytCThresh = ytCThresh;
plt.legend();
#plt.title("Difference in Norm as a function of termination tolerance")
plt.xlabel(r"Iterations");
plt.ylabel(r"$\frac{|\bar{L}^k - L^\star|}{L^\star}$");
#plt.xscale('log')
plt.yscale("log");
plt.grid();
plt.show();
#----------
#Last plot looking at the constrained state's mass density
stateMassPlot, axs =  plt.subplots(1, 2, figsize=(9, 3), gridspec_kw = {'width_ratios':[2, 1]});
# set up state mass figure
axs[0].plot(timeLine, np.sum(optRes[cState,:,:],axis=0), linewidth = 2, label ='unconstrained cvx');
axs[0].plot(timeLine, np.sum(optCRes[cState,:,:],axis=0), linewidth = 2, label ='constrained cvx');
axs[0].set_xlabel(r"Time");
axs[0].set_ylabel(r"Constrained state mass");
axs[0].plot(timeLine, np.sum(finalytCThresh[cState,:,:],axis=0), linewidth = 2, linestyle = "-.", label ='exact penalty FW');
axs[0].grid();
axs[0].legend();
data = {'uCVX': 334413, 'cCVX': 291221, 'eFW': 299457} #FW's is gameObj(ytThresh)
names = list(data.keys())
values = list(data.values())
low = min(values)
high = max(values)
axs[1].grid(zorder=0);
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
axs[1].bar(names, values, zorder=3);

plt.xticks(names, rotation=45)
plt.show();
print "exact penalty game cost with rho = 1: ", gameObj(finalytCThresh, 1)