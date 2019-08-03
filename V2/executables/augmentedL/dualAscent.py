# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 12:56:58 2018

@author: craba
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math as math
import Algorithms.mdpRoutingGame as mrg
import util.mdp as mdp
import Algorithms.dualAscent as da
import util.utilities as ut
import Algorithms.frankWolfe as fw
plt.close('all');
Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);
#----------------set up game -------------------#
sGame = mrg.mdpRoutingGame(seattle,Time, strictlyConvex=True);
seattleGraph=sGame("G");
sGame.setQuad();
numPlayers = 100;
p0 = mdp.resInit(seattleGraph.number_of_nodes(), residentialNum=6./numPlayers);
##---------------set up NOT constrained game -----------------#
print "Solving primal unconstrained case";
optRes, mdpRes = sGame.solve(p0, verbose=False,returnDual=False);

#---------------set up constrained game -----------------#
print "Solving primal constrained case";
cState = 6;   cThresh = 10;                             
sGame.setConstrainedState(cState, cThresh, isLB = True);
optCRes = sGame.solveWithConstraint(p0,verbose = False);
cleanToll = abs(ut.truncate(sGame("optDual")));
cleanToll = np.concatenate((np.array([0,0,0]), cleanToll));
barC = ut.toll2Mat(cState, cleanToll, [sGame.States, sGame.Actions, sGame.Time], True);

#---------------set up dual ascent algorithm-----------------#
lambda0 = np.zeros((sGame.States, sGame.Actions,sGame.Time));
y0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
for i in range(3,Time):
    lambda0[6,:, i] += 600.;

#-------------------- dual ascent --------------------------#
yHist, yCState, lambHist,finalLamb = da.dualAscent(lambda0, y0, p0, sGame.P, 6,10.0, sGame.R, sGame.C, maxErr = 1.0, optVar= barC);

#-------------------- PLOTS --------------------------------#
plt.figure();
Iterations= np.linspace(1,len(yCState), len(yCState));
plt.plot(Iterations, yCState, label = 'density norm');
plt.legend();
plt.grid();
plt.show();

plt.figure();
plt.plot(Iterations,lambHist/la.norm(barC), label = 'dual variable');
plt.ylabel(r"$\frac{\|\tau^k - \tau^\star\|}{\|\tau^\star\|}$");
plt.xlabel("Iterations");
plt.yscale('log'); plt.xscale('log')
#plt.legend();
plt.grid();
plt.show();

def gameObj(yk, length = 1):
    obj = None;
    if length  == 1:
        obj = np.sum(0.5*np.multiply(np.multiply(sGame.R, sGame.R),yk) + np.multiply(sGame.C,yk));
    else:
        obj = np.zeros(length);
        for i in range(length):
            yki = yk[:,:,:,i];
            obj[i] =  np.sum(0.5*np.multiply(np.multiply(sGame.R, sGame.R),yki) + np.multiply(sGame.C,yki));        
    return obj;
a,b,c,d = yHist.shape;
plt.figure();
constrainedCost =gameObj(yHist,d);
averageConstrainedCost = 1.0*constrainedCost;
for i in range(len(constrainedCost)) :
    averageConstrainedCost[i] = np.sum(constrainedCost[0:i])/(i+1);
plt.plot(Iterations,(gameObj(optCRes) - averageConstrainedCost)/gameObj(optCRes));
plt.ylabel(r"$\frac{|L^k - L^\star|}{L^\star}$");
plt.yscale('log');
plt.grid();
plt.show();
#------------------ constraint convergence -----------------------#

#------------------ apply regular penalty -----------------------#
# FW of values converging
threshVal = 1e-3;
def gradF(x):
  return -np.multiply(sGame("reward"), x) + sGame("C") + finalLamb;
x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
ytThresh, ytHist = fw.FW(x0, p0, sGame("probability"), gradF, True, threshVal, maxIterations = 500);
ytHistArr = np.zeros(len(ytHist));
for i in range(len(ytHist)):
#    ytHistArr[i] = la.norm((ytHist[i] - optCRes));
    ytHistArr[i] = gameObj(ytHist[i]);
averagedYt = 1.0*ytHistArr;
for i in range(len(ytHistArr)) :
    averagedYt[i] = np.sum(ytHistArr[0:i])/(i+1);
fig = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';
#---------------- exact penalty -------------------------#
for j in range(1,5):
    curDelta = 0.5**j+1;
    print curDelta;
    def exactGrad(x):
        grad = -np.multiply(sGame("reward"), x)+ sGame("C");
        for time in range(3,Time):
            xDensity = np.sum(x[cState,:,time]);
            if xDensity< cThresh: # put actual constraint here
                grad[cState,:,time] += curDelta*finalLamb[cState,:,time]
        return grad;
    # This is the not state constrained case
    x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
    ytCThresh, ytCHist = fw.FW(x0, p0, sGame("probability"), exactGrad, True, threshVal, maxIterations = 500);
    ytCHistArr = np.zeros(len(ytCHist));
    for i in range(len(ytCHist)):
#        ytCHistArr[i] = la.norm((ytCHist[i]  - optCRes));
        ytCHistArr[i] = gameObj(ytCHist[i]);
    averagedCYt = 1.0*ytCHistArr;
    for i in range(len(ytCHistArr)) :
        averagedCYt[i] = np.sum(ytCHistArr[0:i])/(i+1);
    plt.plot(np.linspace(1, len(ytCHist),len(ytCHist)), abs(gameObj(optCRes) - averagedCYt)/gameObj(optCRes), linewidth = 2, label = str(curDelta));
    
plt.plot(np.linspace(1, len(ytHist),len(ytHist)), abs(gameObj(optCRes) - averagedYt)/gameObj(optCRes), linewidth = 2, label = r'regular penalty');
plt.legend();
#plt.title("Difference in Norm as a function of termination tolerance")
plt.xlabel(r"Iterations");
plt.ylabel(r"$\frac{||y^{\epsilon} - y^{\star}||}{||y^{\star}||}$");
#plt.xscale('log')
plt.yscale("log");
plt.grid();
plt.show();
#----------
# plot the constrained state, and resultant expected cost
timeLine = np.linspace(1,Time,Time)
plt.figure();
plt.plot(timeLine, np.sum(optRes[cState,:,:],axis=0), linewidth = 2, label ='unconstrained cvx');
plt.plot(timeLine, np.sum(optCRes[cState,:,:],axis=0), linewidth = 2, label ='constrained cvx');
plt.plot(timeLine, np.sum(ytThresh[cState,:,:],axis=0), linewidth = 2, linestyle = "-.", label ='regular penalty FW');
plt.xlabel(r"Time");
plt.ylabel(r"Constrained state mass");
#plt.xscale('log')
#plt.yscale("log");
plt.legend();
plt.grid();
plt.show();

fig, axs = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw = {'width_ratios':[2, 1]})
axs[0].plot(timeLine, np.sum(optRes[cState,:,:],axis=0), linewidth = 2, label ='unconstrained cvx');
axs[0].plot(timeLine, np.sum(optCRes[cState,:,:],axis=0), linewidth = 2, label ='constrained cvx');
axs[0].plot(timeLine, np.sum(ytThresh[cState,:,:],axis=0), linewidth = 2, linestyle = "-.", label ='regular penalty FW');
axs[0].grid();
axs[0].set_xlabel('Time')
axs[0].set_ylabel(r"Constrained state mass");
axs[0].legend();
data = {'uCVX': 2914855, 'cCVX': 2871711, 'pFW': 2885634} #FW's is gameObj(ytThresh)
names = list(data.keys())
values = list(data.values())
low = min(values)
high = max(values)
axs[1].grid(zorder=0);
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
axs[1].bar(names, values, zorder=3);

plt.xticks(names, rotation=45)
#fig.autofmt_xdate()
#fig.suptitle('Categorical Plotting')
plt.show()