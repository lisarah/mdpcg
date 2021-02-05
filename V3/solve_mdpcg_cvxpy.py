# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:10:31 2019

@author: craba
"""
import gameSolvers.cvx as cvx

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import util.plot_lib as pt
Time = 20;

sGame = cvx.cvx_solver(20);
#nx.draw(seattleGraph, pos = sGame("graphPos"),with_labels=True);
#plt.show()
#p0 = np.ones((seattleGraph.number_of_nodes()))/seattleGraph.number_of_nodes();
p0 = np.zeros((sGame.G.number_of_nodes()));
#p0[0] = 1.0;
# make all drivers start from residential areas 6 of them
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

print ("Solving primal unconstrained case");
optRes, mdpRes = sGame.solve(p0, verbose=False,returnDual=False);
# mdp.drawOptimalPopulation(Time,
#                           sGame.graphPos,
#                           sGame.G,
#                           optRes,
#                           startAtOne = True,
#                           numPlayers = 60.);
#
cState = 6;                               
sGame.setConstrainedState(cState, 10, isLB = True);
print ("Solving constrained case, state 7 >= 10 case");
optCRes = sGame.solveWithConstraint(p0,verbose = False);
print ("optimal dual: ", sGame.optimalDual)
print ("lower bound" , sGame.stateLB)
print ("upper bound" , sGame.stateUB)
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCRes/10.,
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("lowerBound"));
####
print ("Solving unconstrained problem with new Toll");
optCSol, optCRes = sGame.solve(p0,withPenalty=True,verbose = False, returnDual = False)
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          optCSol/10., 
#                          startAtOne = True,
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("lowerBound"));
#     
# plot constrained state
#timeLine = np.linspace(1,Time,20)
#cTraj = np.sum(optCSol[cState,:,:],axis=0)           
#traj = np.sum(optRes[cState,:,:],axis=0)  
#fig = plt.figure();  
#plt.plot(timeLine,traj,label = "unconstrained trajectory");
#plt.plot(timeLine,cTraj,label = "constrained trajectory"); 
#plt.legend();
#plt.title("State 7 Constrained vs Unconstrained Trajectories")
#plt.xlabel("Time");
#plt.ylabel("Driver Density")
#plt.show();
#yT = optRes[:,:,Time-1];                           
#print "Solving dynamic programming problem of unconstrained problem"; 
#cR = mdp.constrainedReward3D(sGame("reward"),
#                             sGame("optDual") + 0.01, # make dual the interiors
#                             sGame("constrainedState"));         
#tolls = np.concatenate((np.zeros(3),sGame("optDual")));         
#dpVC, dpSolC = dp.dynamicPLinearCost(sGame("reward"),
#                                     sGame("C"),
#                                     sGame("probability"),
#                                     optCSol,
#                                     p0, 
#                                     hasToll = True,
#                                     toll = tolls + 0.1, 
#                                     tollState = cState);
#                                     
                                     
#optTraj_old = np.einsum("ijk->ik", optCSol); 

#    
#mdp.drawOptimalPopulation(Time,
#                          sGame("graphPos"),
#                          sGame("G"),
#                          dpSolC, 
#                          constrainedState = sGame("constrainedState"), 
#                          constrainedUpperBound = sGame("lowerBound"),
#                          # only set is2D to true for dynamic programming
#                          is2D = True, 
#                          startAtOne = True);
                          
timeLine = np.linspace(1,Time,20)
cTraj = np.sum(optCSol[cState,:,:],axis=0)
#dpTraj = dpSolC[cState,:];           
traj = np.sum(optRes[cState,:,:],axis=0)  ;
exampleTraj1 = np.sum(optRes[1,:,:],axis=0);
exampleTraj2 = np.sum(optCSol[1,:,:],axis=0);
start = 0;
#---------- stateTrajectory60PeopleConstrained --------------
fig = plt.figure(); 
plt.plot(timeLine[start+3:], 10*np.ones(20-start-3), label = "constraint", linewidth = 6,color ="r", alpha = 0.3);
plt.plot(timeLine[start:],traj[start:],label = "state 7 unconstrained",linewidth = 2,linestyle = "-.",color ='#1f77b4ff');
plt.plot(timeLine[start:],cTraj[start:],label = "state 7 constrained",linewidth = 2,color ='#1f77b4ff'); 
plt.plot(timeLine[start:],exampleTraj1[start:],label = "state 2 unconstrained",linewidth = 2, linestyle = "-.", color ='#ff7f0eff'); 
plt.plot(timeLine[start:],exampleTraj2[start:],label = "state 2 constrained",linewidth = 2,color ='#ff7f0eff');
#plt.plot(timeLine,dpTraj,label = "dynamic trajectory"); 
plt.grid();
#plt.legend();
#plt.title("Constrained vs Unconstrained Trajectories")
plt.xlabel("Time");
plt.ylabel("Driver Density")
#plt.savefig('test.pdf');
#------------------ toll value ------------------------#
plt.figure();
plt.plot(timeLine[start+3:], sGame.optimalDual, linewidth = 2);
plt.xlabel("Time");
plt.ylabel("Toll Value on State 7");
plt.grid();
#------------- sub plot with toll value and constrained trajectory------------#
blue = '#1f77b4ff';
orange = '#ff7f0eff';
f, (ax1, ax2) = plt.subplots(2, 1, sharex='col');
ax1.plot(timeLine[start+3:], 10*np.ones(20-start-3), label = "constraint", linewidth = 6,color ="r", alpha = 0.3);
ax1.plot(timeLine[start:],traj[start:],label = "state 7 unconstrained",linewidth = 2,linestyle = "-.",color =blue);
ax1.plot(timeLine[start:],cTraj[start:],label = "state 7 constrained",linewidth = 2,color =blue); 
ax1.plot(timeLine[start:],exampleTraj1[start:],label = "state 2 unconstrained",linewidth = 2, linestyle = "-.", color =orange); 
ax1.plot(timeLine[start:],exampleTraj2[start:],label = "state 2 constrained",linewidth = 2,color =orange);
plt.xlabel('Time');
plt.ylabel("Driver Density")
#ax2.plot(timeLine[0:3], np.zeros(3), linewidth = 2, color = blue);
ax2.plot(timeLine[start:], np.concatenate((np.zeros(3),sGame.optimalDual)), linewidth = 2, color = blue);
plt.ylabel("Toll Value")
ax1.grid();
ax2.grid();
plt.show();

##------------------ stateTrajectory60PeopleLegend ------------------------#
plt.figure();
plt.subplot(211)
plt.plot(timeLine[start+3:], 10*np.ones(20-start-3), label = "bound value", linewidth = 6,color ="r", alpha = 0.3);
plt.plot(timeLine[start:],traj[start:],label = r"$s_7$ unconstrained",linewidth = 2,linestyle = "-.",color =blue);
plt.plot(timeLine[start:],cTraj[start:],label = r"$s_7$ constrained",linewidth = 2,color =blue); 
plt.plot(timeLine[start:],exampleTraj1[start:],label = r"$s_2$ unconstrained",linewidth = 2, linestyle = "-.", color =orange); 
plt.plot(timeLine[start:],exampleTraj2[start:],label = r"$s_2$ constrained",linewidth = 2,color =orange);
plt.plot(timeLine[start:],np.concatenate((np.zeros(3),sGame.optimalDual))/60., label = "toll charged", linestyle = ":", linewidth = 2, color = 'k');
plt.grid();
plt.xlabel('Time'); 
plt.ylabel('Density');  
plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left',fontsize = 'x-small');
plt.show();

plt.plot()


#--------------------- heat map ----------------------------------------------#
diffRes = optRes - optCSol;
diffResPos = np.where(diffRes > 0, diffRes, 0);
diffResNeg = np.where(diffRes < 0, diffRes, 0);
stateRes = np.einsum('ijt->it', diffResPos);
stateResPos = np.einsum('ijt->it', diffResPos);
stateResNeg = np.einsum('ijt->it', diffResNeg);
timeAveragePos = np.einsum('ij->i', abs(stateResPos));
timeAverageNeg = np.einsum('ij->i', abs(stateResNeg));
timeAverage = np.einsum('ij->i', abs(stateRes));
plt.figure();
width = 17;
fullW = 20;
#for node in sGame("graphPos"):
#    if timeAveragePos[node-1] > timeAverageNeg[node - 1]:
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size=fullW*timeAveragePos[node-1], nodelist = [node], node_color='r',alpha = 1);
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size= width*timeAveragePos[node-1], nodelist = [node], node_color='w',alpha = 1);
#        
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size=fullW*timeAverageNeg[node-1], nodelist = [node], node_color='b',alpha = 1);
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size= width*timeAverageNeg[node-1] - width, nodelist = [node], node_color='w',alpha = 1);
#    else:
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size=fullW*timeAverageNeg[node-1], nodelist = [node], node_color='b',alpha = 1);
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size= width*timeAverageNeg[node-1], nodelist = [node], node_color='w',alpha = 1);
#        
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size=fullW*timeAveragePos[node-1], nodelist = [node], node_color='r',alpha = 1);
#        nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size= width*timeAveragePos[node-1], nodelist = [node], node_color='w',alpha = 1);
#        
#    nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size=20*timeAverageNeg, node_color='r',with_labels=True, font_weight='bold',alpha = 0.8);
#nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size=20*timeAverageNeg, node_color='r',with_labels=True, font_weight='bold',alpha = 0.8, facecolors = 'None');
#nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size=20*timeAveragePos, node_color='c',with_labels=True, font_weight='bold',alpha = 0.5, facecolors = 'None');
#nx.draw(sGame("G"),sGame("graphPos"),node_size=5*timeAverageNeg,node_color='r',with_labels=True, font_weight='bold',alpha = 1);
#nx.draw_networkx_nodes(sGame("G"), pos=sGame("graphPos"), node_size= 500, node_color=timeAveragePos,alpha = 0.8, cmap=plt.cm.Blues);
nx.draw_networkx_nodes(sGame.G, pos=sGame.graphPos, node_size= 500, node_color=timeAverageNeg,alpha = 0.8, cmap=plt.cm.Reds);
nx.draw(sGame.G,sGame.graphPos,node_size=1,node_color='none',with_labels=True, font_weight='bold',alpha = 0.8)  
plt.show();    
    