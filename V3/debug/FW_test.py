# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:24:17 2019

@author: sarahli
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:06:00 2018

@author: craba
"""
import gameSolvers.mdpcg as mdpcg
#import util.utilities as ut
import algorithm.FW as fw

import numpy as np
#import matplotlib.pyplot as plt

Time = 20;

sGame = mdpcg.game(Time);
seattleGraph=sGame.G;
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


# FW of values converging
threshVal = 1e-3;
def gradF(x):
    return np.multiply(sGame.R, x) + sGame.C;

def obj(y):
    objTens = 0.5*np.multiply(np.multiply(sGame.R, y), y) + np.multiply(sGame.C,y);
    return np.sum(objTens);
#-------------------- With regular Penalty -------------------#
x0 = np.zeros((sGame.States, sGame.Actions, sGame.Time));   
ytThresh, ytHist = fw.FW(x0, p0,sGame.P, gradF, isMax = False, returnHist =False);

print ("Objective is ", obj(ytThresh))
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

