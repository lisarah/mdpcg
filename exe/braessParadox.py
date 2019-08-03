# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 09:23:26 2018

@author: craba
"""

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


import numpy as np
import numpy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
plt.close('all')
Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);

socialGame = mrg.mdpRoutingGame(seattle,Time);
socialGame.setQuad();

sGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=sGame("G");
sGame.setQuad();
#nx.draw(seattleGraph, pos = sGame("graphPos"),with_labels=True);
#plt.show()
#p0 = np.ones((seattleGraph.number_of_nodes()))/seattleGraph.number_of_nodes();
p0 = np.zeros((seattleGraph.number_of_nodes()));
numPlayers = 1000;
#p0[0] = 1.0*numPlayers;
# make all drivers start from residential areas 6 of them
residentialNum = 6;
residentialList = [2,3,7,8,10,11];
p0[2] = 1.*numPlayers/residentialNum;
p0[3] = 1.*numPlayers/residentialNum;
p0[7] = 1.*numPlayers/residentialNum;
p0[8] = 1.*numPlayers/residentialNum;
p0[10] =1.*numPlayers/residentialNum;
p0[11] =1.*numPlayers/residentialNum;
#Iterations= 10;
#players = np.zeros(Iterations);
#optDifference = np.zeros(Iterations);
#optSocialObj = np.zeros(Iterations); optObj = np.zeros(Iterations);
#optSocialRes = np.zeros((sGame.States,sGame.Actions,Time, Iterations));
#optRes = np.zeros((sGame.States,sGame.Actions,Time, Iterations));
#for people in range(Iterations):
#    numPlayers = people*200 + 2000;
##    numPlayers = 3700;
#    players[people] = numPlayers;
#    p0[2] = 1.*numPlayers/residentialNum;
#    p0[3] = 1.*numPlayers/residentialNum;
#    p0[7] = 1.*numPlayers/residentialNum;
#    p0[8] = 1.*numPlayers/residentialNum;
#    p0[10] =1.*numPlayers/residentialNum;
#    p0[11] =1.*numPlayers/residentialNum;
#
#    socialGame = mrg.mdpRoutingGame(seattle,Time);
#    socialGame.setQuad();    
#    sGame = mrg.mdpRoutingGame(seattle,Time);
#    sGame.setQuad();
#
#    print ("-------------People = ", numPlayers, " -------------------------");
#    print ("Solving unconstrained social optimal");
#    optSocialRes[:,:,:,people], optSocialObj[people] = socialGame.solve(p0, verbose=False,returnDual=False, isSocial = True );
#    print ("Solving unconstrained user optimal");
#    optRes[:,:,:,people], optObj[people] = sGame.solve(p0, verbose=False,returnDual=False, isSocial = False );
#    optObj[people] = sGame.socialCost(optRes[:,:,:,people]);
#    optDifference[people] = la.norm(optSocialRes[:,:,:,people] - optRes[:,:,:,people]);
#    
#    
#    
##    print ("------------ Solving for constrained user Optimal -------------------------");
##    optDiff = ut.truncate(optSocialRes[:,:,:,people] - optRes[:,:,:,people], 1e1);
##    nonZeroDiff = ut.nonZeroEntries(optDiff);
##    constraintList = ut.constraints(nonZeroDiff, optSocialRes, optRes);
##    optCRes = sGame.solveWithConstraint(p0,verbose = False, constraintList = constraintList);
##    optDual = ut.matGen(constraintList, sGame.optimalDual, [sGame.States, sGame.Actions, Time]);
##    optObj[people] = sGame.socialCost(optCRes);
##    print ("Social cost at constrained User Optimal");mming problem of unconstrained problem"; 
#cR = mdp.constrainedReward3D(sGame("reward"),
#                             sGame("optDual") + 0.01, # make dual the interiors
#                             sGame("constrainedState"));         
#tolls = np.concatenate((np.zeros(3),sGame("optDual")));         
#dpVC, dpSolC = dp.dynamicPLinearCost(sGame("reward"),
#                               
##    print (optObj[people]);
#    
##timeLine = np.linspace(1,Time,20)
#fig = plt.figure();  
#plt.plot(players,optSocialObj,linewidth = 2, label = "social objective");
#plt.plot(players,optObj,linewidth = 2, label = "user objective"); 
#plt.legend();
##plt.title("Braess Paradox occurence as function of time")
#plt.xlabel("Number of ride share drivers in game");
#plt.ylabel("Total objective value");
#plt.grid();
#plt.show();
#yT = optRes[:,:,Time-1];                           
#print ("Solving dynamic programming problem of unconstrained problem"); 
#    
#fig = plt.figure();  
#plt.plot(players,optDifference,label = "difference in policies");
#plt.legend(fontsize = 'x-small');
##plt.title("Braess Paradox occurence as function of time")
#plt.xlabel("Number of ride share drivers in game");
#plt.ylabel("Total policy difference in 2 norm");
#plt.grid();
#plt.show();   

#------------------- Set number of Players, vary toll --------------------------------#
Iterations = 10;
numPlayers = 3500;
p0[2] = 1.*numPlayers/residentialNum;
p0[3] = 1.*numPlayers/residentialNum;
p0[7] = 1.*numPlayers/residentialNum;
p0[8] = 1.*numPlayers/residentialNum;
p0[10] =1.*numPlayers/residentialNum;
p0[11] =1.*numPlayers/residentialNum;
tolls = np.zeros(Iterations);
tolledObj = np.zeros(Iterations);
constraintNumber = np.zeros(Iterations);
socialGame = mrg.mdpRoutingGame(seattle,Time);
socialGame.setQuad();    
sGame = mrg.mdpRoutingGame(seattle,Time);
sGame.setQuad();

print ("-------------People = ", numPlayers, " -------------------------");
print ("Solving unconstrained social optimal");
optSocialRes, optSocialObj = socialGame.solve(p0, verbose=False,returnDual=False, isSocial = True );
print ("Solving unconstrained user optimal");
optRes, optObj = sGame.solve(p0, verbose=False,returnDual=False, isSocial = False );
optObj = sGame.socialCost(optRes);

optDifference = la.norm(optSocialRes- optRes);
    
plotOut = np.zeros((4, Iterations)); # tau_min, tau_max, \sum(y*tau)    
for incre in range(Iterations):
    tolls[incre] = 100.8 - incre*10;    
    print ("------------ Solving for constrained user Optimal -------------------------");
    optDiff = ut.truncate(optSocialRes- optRes, tolls[incre]);
    nonZeroDiff = ut.nonZeroEntries(optDiff);
    constraintList = ut.constraints(nonZeroDiff, optSocialRes, optRes);
    constraintNumber[incre] = len(constraintList);
    optCRes = sGame.solveWithConstraint(p0,verbose = False, constraintList = constraintList);
    optDual = ut.matGen(constraintList, sGame.optimalDual, [sGame.States, sGame.Actions, Time]);
    tolledObj[incre] = sGame.socialCost(optCRes); # +  tolls[incre]* np.sum(np.multiply(optDual, optCRes));
    plotOut[0,incre] = np.min(optDual);
    plotOut[1,incre] = np.max(optDual);
    optDualNeg = np.where(optDual < 0., optDual, 0.);
    optDualPos = np.where(optDual > 0., optDual, 0.);
    plotOut[2,incre] = np.sum(np.multiply(optDualNeg, optCRes));
    plotOut[3,incre] = np.sum(np.multiply(optDualPos, optCRes));
    print ("Social cost at constrained User Optimal");
    print (tolledObj[incre]);
#    
fig = plt.figure();  
plt.plot(constraintNumber,optSocialObj*np.ones(Iterations), color = 'r', linewidth = 5, label = "social objective", alpha = 0.3);
plt.plot(constraintNumber,optObj*np.ones(Iterations), color = 'k', linewidth = 5, label = "unconstrained user objective", alpha = 0.3);
plt.plot(constraintNumber,tolledObj, linewidth = 2, color ='#1f77b4ff', label = "constrained user objective"); 
plt.legend(fontsize = 'x-small');
#plt.yscale('log');
#plt.xlim((10, 400))   # set the xlim to xmin, xmax
#plt.title("Constrained User Optimal with Decreasing Toll Precision")
plt.xlabel("Number of constraints imposed");
plt.ylabel("Total objective value");
plt.grid();
minorLocator = tck.MultipleLocator(5);
plt.show();    
#

figTolls = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';
plt.plot(constraintNumber, abs(plotOut[0,:]), color = blue, label = r"$|\min(\tau__{tsa})|$" );
plt.plot(constraintNumber, plotOut[1,:], color = orange, label = r"$|\max(\tau__{tsa})|$" );   
#plt.plot(constraintNumber, plotOut[2,:]/3500., color = 'k', label = r"$\frac{\sum_{tsa} y_{tsa}\tau_{tsa}}{\sum_{0sa}y_{0sa}}$" );    
plt.xlabel("Number of constraints imposed");
plt.ylabel("Toll Value");
plt.legend();
plt.grid();
minorLocator = tck.MultipleLocator(5);
plt.show();

figNetIncome = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';  
plt.plot(constraintNumber, abs(plotOut[2,:])/optSocialObj, color = blue, label = r"$|\sum_{tsa} y_{tsa}(\tau_{tsa})_-|$" );    
plt.plot(constraintNumber, plotOut[3,:]/optSocialObj, color = orange, label = r"$\sum_{tsa} y_{tsa}(\tau_{tsa})_+$" );    
plt.plot(constraintNumber, (abs(plotOut[2,:]) - plotOut[3,:])/optSocialObj, color = 'k', label = r"$\sum_{tsa} y_{tsa}\tau_{tsa}$" );    
plt.xlabel("Number of constraints imposed");
plt.ylabel("Net incentives/tolls")
plt.legend();
plt.grid();
plt.show();

plt.figure();
plt.subplot(211)
plt.plot(constraintNumber, abs(plotOut[2,:])/optObj, color = blue, label = r"$\frac{h_{driv}}{J(x^\star)}$" );    
plt.plot(constraintNumber, plotOut[3,:]/optObj, color = orange, label = r"$\frac{h_{plan}}{J(x^\star)}$" );    
plt.plot(constraintNumber, (abs(plotOut[2,:]) - plotOut[3,:])/optObj, color = 'k', label = r"$\frac{h_{net}}{J(x^\star)}$" );    
plt.grid();
plt.xlabel('Number of constraints imposed');   
plt.ylabel('Income/loss');
plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left',fontsize = 'x-small');
plt.show();

#fig = plt.figure();  
#plt.plot(tolls,constraintNumber,label = "Number of constraints");
#plt.legend(fontsize = 'x-small');
##plt.title("Tolling state action pairs as a function of toll sensitivity")
#plt.xlabel("Sensitivity of Tolling Threshold");
#plt.ylabel("Total number of tolls imposed")
#plt.show();    
#    
