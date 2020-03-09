# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:09:00 2019

@author: craba
"""
import gameSolvers.mdpcg as mdpcg
import algorithm.FW as FW
import numpy as np
from mystic.solvers import fmin_powell
from mystic.monitors import VerboseMonitor
import gc

class bilevel(mdpcg.game):
    states = None;
    actions = None;
    Time = None;
    p0 = None;
    y0 = None;
    file = open("debug.txt", "a+"); 
    fileIter = 0; 
    maxfileOutput = 100;
    def setP0(self, p0):
        self.p0 = 1.0*p0;
        
    def setDimensions(self):
        self.states, self.actions,self.Time = self.R.shape;
        
    def p02Prime(self,p0):
        primeY = np.zeros((self.states, self.actions, self.Time));
        for i in range(self.states):
            primeY[i, 0, 0] = p0[i]; 
        self.y0 = primeY;
        return self.vecPrime(primeY, np.zeros((self.states, self.actions, self.Time)));
    
    def tensPrime(self, vector):
        prime = np.reshape(vector, (2, self.states, self.actions, self.Time));
        y = prime[0,:,:,:];
        eps = prime[1,:,:,:];   
        return y, eps;
    
    def vecPrime(self, tens1, tens2):
        return np.reshape(np.array([tens1, tens2]), (2*self.states*self.actions*self.Time));
    
    def obj(self,primeV):
        y, eps = self.tensPrime(primeV);
        objVal = 0.5*np.multiply(np.multiply(self.R, y),y) + np.multiply(self.C, y);
        return np.linalg.norm(eps)*1e3 + np.sum(objVal);

    def constraints(self, primeV):
        y, eps = self.tensPrime(primeV);
        self.C = self.C + eps;
#        print ("in constraints");
        newY = self.frankWolfe(self.p0);
        self.fileIter += 1;
        if self.fileIter % self.maxfileOutput == 0:
            gc.collect();           
            self.file.write("Evaluating constraint %d\r\n" % (self.fileIter));
            print ("Evaluating constraint, ", self.fileIter);
        self.C = self.C - eps;    
        return self.vecPrime(newY, eps);
    
    def gradF(self, y):
        return np.multiply(self.R, y) + self.C;
    
    def frankWolfe(self, p0):
        yijt, hist = FW.FW(self.y0, p0, self.P, self.gradF,returnHist = False);
        yTens = np.reshape(yijt, (self.states, self.actions, self.Time));    
        return yTens;
    
    def solve(self, p0):
        self.setP0(p0);
        self.setDimensions();
        prime = self.p02Prime(p0);
        stepMon = VerboseMonitor(1);
        solution = fmin_powell(self.obj, prime, constraints = self.constraints, itermon = stepMon,maxiter= 2);
        self.file.close();

        return self.tensPrime(solution);
    
class mysticGame(mdpcg.game):
    states = None;
    actions = None;
    Time = None;
    p0 = None;
    def setP0(self, p0):
        self.p0 = 1.0*p0;
    def setDimensions(self):
        self.states, self.actions,self.Time = self.R.shape;
    def p02yijt(self,p0):
        yijt = np.zeros((self.states, self.actions,self.Time));
        for i in range(self.states):
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
        return solution;
    
