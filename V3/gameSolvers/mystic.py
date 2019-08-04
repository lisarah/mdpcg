# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:09:00 2019

@author: craba
"""
import mdpcg as mdpcg
import numpy as np
from mystic.solvers import fmin_powell
from mystic.monitors import VerboseMonitor

class bilevel(mdpcg.game):
    states = None;
    actions = None;
    Time = None;
    p0 = None;
    def setP0(self, p0):
        self.p0 = 1.0*p0;
    def setDimensions(self):
        self.states, self.actions,self.Time = self.R.shape;
    def p02yijt(self,p0):
        states,actions, Time = self.R.shape;
        yijt = np.zeros((states, actions, Time));
        for i in range(states):
            yijt[i, 0, 0] = p0[i];
        
        return yijt;
    def obj(self, yijt):
        yTens = np.reshape(yijt, (self.states, self.actions, self.Time));
        F = 0.5*np.multiply(np.multiply(yTens, self.R), yTens) \
            + np.multiply(self.C, yTens);
        return np.sum(F);
    
    def constraints(self, yijt):
        yTens = np.reshape(yijt, (self.states, self.actions, self.Time));
        # positivity
        for i in range(self.states):
            for j in range(self.actions):
                for t in range(self.Time):
                    if yTens[i,j,t] < 0:
                        yTens[i,j,t] = 0;
        # initial condition
        for i in range(self.states):
            initSum = sum([yTens[i,j,0] for j in range(self.actions)]);
            if initSum < 1e-10:
                yTens[i,j,0] = self.p0[i]/self.actions;
            else:
                for j in range(self.actions):
                    yTens[i,j,0] = self.p0[i]/initSum*yTens[i,j,0];
            

        # mass conservation
        for i in range(self.states):
            for t in range(1,self.Time):
                prevSum = np.sum(np.multiply(self.P[i,:,:], yTens[:,:, t-1]));
                curSum = sum([yTens[i,j,t] for j in range(self.actions)]);
                if prevSum != curSum:
                    if curSum > 0:
                        for j in range(self.actions):
                            yTens[i,j,t] = prevSum/curSum*yTens[i,j,t];  
                    else:
                        for j in range(self.actions):
                            yTens[i,j,t] = prevSum/self.actions;
        return np.reshape(yTens, [self.states*self.actions*self.Time]);  

    def solve(self, p0): 
        self.setP0(p0);
        self.setDimensions();

        y0 = self.p02yijt(p0) ;        
        stepMon = VerboseMonitor(1);
        solution = fmin_powell(self.obj, np.reshape(y0,[self.states*self.actions*self.Time]), constraints = self.constraints, itermon = stepMon)
        print (solution);