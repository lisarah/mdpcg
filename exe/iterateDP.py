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

Time = 20;
seattle = mrg.gParam("seattleQuad", None, None);

sGame = mrg.mdpRoutingGame(seattle,Time);
seattleGraph=sGame("G");
sGame.setQuad();
#nx.draw(seattleGraph, pos = sGame("graphPos"),with_labels=True);
#plt.show()
#p0 = np.ones((seattleGraph.number_of_nodes()))/seattleGraph.number_of_nodes();
p0 = np.zeros((seattleGraph.number_of_nodes()));
numPlayers = 100;
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

print "Solving primal unconstrained case";
optRes, optObj = sGame.solve(p0, verbose=False,returnDual=False);
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optRes,
#                          startAtOne = True,
#                          numPlayers= p0[0]);

cleanOpt = abs(ut.truncate(optRes));
V, VTotal, yt = dp.iterativeDP(sGame.States,
                               sGame.Actions,
                               sGame.Time, 
                               numPlayers,
                               p0,
                               sGame("reward"),
                               sGame("C"),
                               sGame("probability"));
                               
print"----------    Dynamic Programming value     --------------";
print VTotal;
yDP = abs(ut.truncate(yt));
totalDiff = la.norm(ut.truncate(abs(yDP - cleanOpt)))
print "Maximum difference   ", (ut.truncate(abs(yDP - cleanOpt))).max()/numPlayers;
    

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
    optimalDual = np.concatenate((np.zeros(3), sGame("optDual") ));
else:
    optimalDual = -np.concatenate((np.zeros(3), sGame("optDual")));
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

V, VTotal, yt = dp.iterativeDP(sGame.States,
                               sGame.Actions,
                               sGame.Time, 
                               numPlayers,
                               p0,
                               sGame("reward"),
                               sGame("C"),
                               sGame("probability"),
                               hasToll = True,
                               toll = optimalDual + 1.0,
                               tollState = cState);
                               
print"----------    Dynamic Programming value     --------------";
print VTotal- np.sum(optimalDual*cThresh);
yDP = abs(ut.truncate(yt));
totalDiff = la.norm(ut.truncate(abs(yDP - cleanOpt)))
print "Maximum difference   ", (ut.truncate(abs(yDP - cleanOpt))).max()/numPlayers;
                   
timeLine = np.linspace(1,Time,20)
cTraj = np.sum(optCSol[cState,:,:],axis=0)
dpTraj =  np.sum(yDP[cState,:,:],axis=0)       
traj = np.sum(optRes[cState,:,:],axis=0)  
fig = plt.figure();  
plt.plot(timeLine,traj,label = "unconstrained trajectory");
plt.plot(timeLine,cTraj,label = "constrained trajectory"); 
plt.plot(timeLine,dpTraj,label = "dynamic trajectory"); 
plt.legend();
plt.title("State 7 Constrained vs Unconstrained Trajectories")
plt.xlabel("Time");
plt.ylabel("Driver Density")
plt.show();
yT = optRes[:,:,Time-1];                           
print "Solving dynamic programming problem of unconstrained problem"; 

