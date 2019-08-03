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
import mdpRoutingGame as mrg
import mdp as mdp
import dynamicProgramming as dp



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
optRes = sGame.solve(p0, verbose=False,returnDual=False);
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optRes,
#                          startAtOne = True,
#                          numPlayers= p0[0]);



cState = 7;
timeLine = np.linspace(1,Time,20)
#cTraj = np.sum(optCSol[cState,:,:],axis=0)
#dpTraj =  np.sum(yDP[cState,:,:],axis=0)       
traj = np.sum(optRes[cState,:,:],axis=0)  
fig = plt.figure();  
plt.plot(timeLine,traj,label = "unconstrained trajectory");
#plt.plot(timeLine,cTraj,label = "constrained trajectory"); 
#plt.plot(timeLine,dpTraj,label = "dynamic trajectory"); 
plt.legend();
plt.title("State 7 Constrained vs Unconstrained Trajectories")
plt.xlabel("Time");
plt.ylabel("Driver Density")
plt.show();
yT = optRes[:,:,Time-1];                           
print "Solving dynamic programming problem of unconstrained problem"; 

