# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:03:22 2018

@author: craba
"""
import numpy as np
import math 
import Algorithms.frankWolfe as fw

def ALM(c0, mu0, f0, p0, R, C, P, Cap, maxErr = 0.5):
    states, actions, time = f0.shape;
    maxIterations = 20;
    it = 1;
    err = 1000.;
    kappa = 1.5; beta = 0.8
    fk = f0; ck = c0; muk = mu0;
    lastNorm = np.linalg.norm(Cap - fk);
    errRes = np.zeros((maxIterations));

    while it <= maxIterations and err >= maxErr:
        #--- gradient is kth iterative coefficient dependent ---#
        def gradF(xk):
            g = np.multiply(R, xk) - C ;
            for ind in range(len(muk)):
                sap = muk[ind].index;
    #            print sap;
                if muk[ind].upperBound:
                    if ck*(xk[sap] - Cap[sap]) + muk[ind].value > 0:
                        g[sap] += ck*(xk[sap] - Cap[sap]) + muk[ind].value;
                else: 
                    if ck*(Cap[sap] - xk[sap]) + muk[ind].value > 0:
                        g[sap] += -(ck*(Cap[sap] - xk[sap]) + muk[ind].value);
            return g; 
        #------------ solve sub problem ------------#
        fNext, fHist = fw.FW(fk, p0, P, gradF, True, maxErr);
        print "updated flow difference", np.linalg.norm(fk - fNext);
        fk = fNext;
        #------------ multiplier update ------------#
        for ind in range(len(mu0)):
            sap = mu0[ind].index;
            if muk[ind].upperBound:
                if fk[sap] < Cap[sap] - muk[ind].value/ck:
                    newVal = muk[ind].value + ck *(fk[sap] - Cap[sap]);
                else:
                    newVal = 0;
            else: 
                if fk[sap] > muk[ind].value/ck + Cap[sap]:
                    newVal =  muk[ind].value + ck * (Cap[sap] - fk[sap] );
                else: 
                    newVal = 0;
#            if newVal > 0:
#                print " new value update: ", newVal;
            muk[ind]= muk[ind]._replace(value=newVal);

        #------------ step size update ------------#
        newNorm = np.linalg.norm(Cap - fk);
        print newNorm;
        if newNorm >= beta*lastNorm:
            ck = kappa*ck;
        lastNorm  = newNorm;

        #------------ error update -----------------#
        err = 0;
        for ind in range(len(mu0)):
            sap = muk[ind].index;
            err += (Cap[sap] - fk[sap])**2;
        errRes[it-1] = math.sqrt(err);
        print " ------------ ALM Iteration ", it, " -----------";
        print "error = ", err;
        print "new penalty constant ", ck;
        it += 1;

    print " ------------ Augmented Lagrangian summary -----------";
    print "number of iterations = ", it;
    print "total error in cost function = ", err;
    return muk, errRes, fk;
