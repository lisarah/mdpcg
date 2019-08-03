# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:30:52 2018

@author: craba
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:10:25 2018

@author: sarah
"""
import Algorithms.infMDP as imdp
import numpy as np
import Algorithms.frankWolfe as fw
import matplotlib.pyplot as plt
seattle = imdp.gParam("seattleQuad", None, None);
sGame = imdp.infMDP(seattle, beta = 0.8);
seattleGraph=sGame.G;
#nx.draw(seattleGraph, pos = sGame("graphPos"),with_labels=True);
#plt.show()
#p0 = np.ones((seattleGraph.number_of_nodes()))/seattleGraph.number_of_nodes();
p0 = np.zeros((seattleGraph.number_of_nodes()));
#p0[0] = 1.0;
# make all drivers start from residential areas 6 of them
residentialNum = 0.1;
p0[2] = 1./residentialNum;
p0[3] = 1./residentialNum;
p0[7] = 1./residentialNum;
p0[8] = 1./residentialNum;
p0[10] = 1./residentialNum;
p0[11] = 1./residentialNum;

discountF = 0.8;#np.linspace(0.0,1.0,num=20);

#infRes = np.zeros([sGame.States, sGame.Actions, Iterations]);
#infObj = np.zeros(Iterations);
cState = 6;
print "Solving for discount factor: ", discountF;
sGame = imdp.infMDP(seattle, beta =  discountF);
infObj, infRes= sGame.solve(p0,verbose=False,returnDual=False);
    #mdp.drawOptimalPopulation(Time,
    #                          sGame("graphPos"),
    #                          sGame("G"),
    #                          optRes/10.,
    #                          startAtOne = True);
#------------------- FW on Infinite Horizon ------------------------------
print "solving with FW";
threshVal = 1e-3;
def gradF(x):
  return -np.multiply(sGame("reward"), x) + sGame("C");   
  
x0 = np.zeros((sGame.States, sGame.Actions));   
ytThresh, ytHist = fw.FW(x0, p0, sGame("probability"), gradF, True, threshVal);
ytHistArr = np.zeros(len(ytHist));
for i in range(len(ytHist)):
    ytHistArr[i] = np.linalg.norm((ytHist[i] - infRes));
#fig = plt.figure();
blue = '#1f77b4ff';
orange = '#ff7f0eff';                              
plt.plot(np.linspace(1, len(ytHist),len(ytHist)), ytHistArr, linewidth = 2, label = r'$||\cdot||_2$',color = blue);
plt.show();
#------------------- Plot 1: Changes for state 7 ------------------------------
#fig = plt.figure(); 
#cPop = np.zeros(Iterations); 
#for discount in range(Iterations):
#    cPop[discount] = np.sum(infRes[cState,:, discount]) ; 
#
#plt.plot(discountF, cPop,label =(r'df: %0.2f'%(discountF[i])));                      
#
#plt.legend();
#plt.title("State 7 Trajectory for different discount factors (60 people game)")
#plt.xlabel("Time");
#plt.ylabel("Population Density")
#plt.show();
#
###------------------- Plot 2: Total value of income ----------------------------
#fig = plt.figure();  
#plt.plot(discountF,infObj);                      
#plt.title("Total income earned as a function of discount factor")
#plt.xlabel("Discount Factor");
#plt.ylabel("Total Objective Value")
#plt.show();
