# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:22:43 2018

@author: craba
"""

import Algorithms.mdpRoutingGame as mrg
import util.mdp as mdp
import util.fixedDemandMDP as fdm
import Algorithms.dynamicProgramming as dp
import util.utilities as ut
import Algorithms.frankWolfe as fw
import Algorithms.projectGradientMethod as pgm

import numpy as np
import numpy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
import cvxpy as cvx
#------------ Definitions --------------------------------#
isFixed = False;
Time = 20;
nodes = 13; edges = 25;
ER, P, S, Sc, D,Dc = fdm.erdosRenyiMDP(nodes, edges);
graphPos = nx.circular_layout(ER);
n,m = S.shape; states = n; actions = m;
STensor = np.reshape(np.tile(S, Time),(n,m,Time),'F');
ScTensor =  np.reshape(np.tile(Sc,Time),(n,m,Time),'F');
#DTensor =  np.reshape(np.tile(D,Time),(n,m,Time),'F');
#DcTensor = np.reshape(np.tile(Dc,Time),(n,m,Time),'F');
nx.draw(ER, node_color='c', edge_color='k',
                font_weight='bold', transparent=False) # With specified positions
#nx.draw(ER);
plt.show();
#demand = 10.* mdp.resInit(nodes);
#Prand = P[:,:,1]; # always take action 1
#C0 = np.zeros((states, actions,Time));
#R = STensor;
#C = ScTensor;
#trajectory = np.zeros((states, actions,Time));
## propagate the population density using  action 1      
#for t in range(Time):
#    pol = 1;
#    population = np.zeros(len(demand));
#    if t == 0:
#        population= demand;
#    else:        
#        population = np.einsum('ijk,jk',P,trajectory[:,:,t-1]);
#    for s in range(states):
#        trajectory[s, pol, t] = population[s];
# 
#x0 = 1.0*trajectory; z0 = np.zeros(len(demand));       
#
#    
#def gradF(y, z):
#    grad = np.multiply(STensor, y) + ScTensor ;
#    if isFixed:
#        return grad, np.zeros(len(z));
#    else:
#        return grad, (np.multiply(D, demand - z) + Dc);
#    
#
#
#def singleGradF(y):
#    grad = np.multiply(STensor, y) + ScTensor ;
#    return grad;
#
#def objVal(xtsa):
#    obj = 0.5*np.multiply(np.multiply(xtsa,xtsa),STensor) + np.multiply(xtsa,ScTensor);
#    scalarObj = np.sum(obj);
#    if isFixed ==False:
#        ps = sum([xtsa[:,action, 0] for action in range(m)]);
#        obj2= -0.5*np.multiply(np.multiply(D,ps), ps) - np.multiply(ps,Dc);
#        scalarObj += np.sum(obj2);
#        
#    return scalarObj;
#
#print "-------- Solve using CVX -----------" 
## set up variable
#y_ijt = {};
#if isFixed == False:
#    ps = {};
#for i in range(states):
#    if isFixed == False:
#        ps[(i)] = cvx.Variable();
#    for j in range(actions):
#        for t in range(Time):
#            y_ijt[(i,j,t)] = cvx.Variable();
#            
#objF = sum([sum([sum([0.5*cvx.pos(STensor[i,j,t])*cvx.square(y_ijt[(i,j,t)])
#             for i in range(states) ]) 
#        for j in range(actions)]) 
#   for t in range(Time)]) \
#       + sum([sum([sum([(ScTensor[i,j,t])*y_ijt[(i,j,t)]
#             for i in range(states) ]) 
#        for j in range(actions)]) 
#   for t in range(Time)]);
#if isFixed == False:
#    objF+=  -(sum([-0.5*cvx.pos(D[i])*cvx.square(ps[(i)]) for i in range(states)]) \
#            + sum([(Dc[i])*ps[(i)] for i in range(states) ])) ;  
#       
## constraints enforcement
#positivity = [];
#for i in range(states):                
#    for t in range(Time):                     
#        for j in range(actions):
#            # positivity constraints
#            positivity.append(y_ijt[(i,j,t)] >= 0.);
#            
#initialCondition = [];
#playGameCondition = [];
#for i in range(states):
#    # Enforce initial condition
#    initState = sum([y_ijt[(i,j,0)] for j in range(actions)]);
#    if isFixed:
#        initialCondition.append(initState == demand[i]); 
#    else:
#        playGameCondition.append(initState == ps[(i)]);
#        initialCondition.append(initState <= demand[i]); 
#
#massConservation = [];
#for i in range(states):
#    for t in range(Time-1):  
#        # mass conservation constraints between timesteps
#        prevProb = sum([sum([y_ijt[(iLast,j,t)]*P[i,iLast,j] 
#                        for iLast in range(states) ]) 
#                   for j in range(actions)]) ;
#        newProb = sum([y_ijt[(i,j,t+1)] 
#                  for j in range(actions)]);
#        massConservation.append(newProb == prevProb); 
#        
#if isFixed:
#    groundTruth = cvx.Problem(cvx.Minimize(objF), 
#                              positivity+massConservation+initialCondition);   
#else:
#    groundTruth = cvx.Problem(cvx.Minimize(objF), 
#                             positivity+massConservation+
#                             playGameCondition + initialCondition);
#                          
#objTrue= groundTruth.solve(solver=cvx.ECOS, verbose=True);
#print objTrue;
#optRes = mdp.cvxDict2Arr(y_ijt,[states,actions,Time]);   
#groundCost = np.multiply(STensor, optRes) + ScTensor; 
#if isFixed ==False:                       
#    optCommitment =  mdp.cvxDict2Arr(ps,[states]);      
#states = n;
#                       
#print "------ Frank Wolfe Solver -----------"
#if isFixed:
#    xStar, history = fw.FW(x0,1.0*demand, P, singleGradF, 0.01);
#else: 
#    xStar, zStar, history = fw.FW_fixedDemand(x0,z0, 1.0*demand, P, gradF, 0.01);
###------------ FW only ------- # 
#xK = np.zeros(len(history));
#for xk in range(len(history)):
#    xK[xk] = objVal(history[xk]);
##plt.figure();  
##plt.plot(abs((xK - objTrue)/objTrue),label = "running average difference");
##plt.yscale('log');
##plt.legend();
##plt.grid();
##plt.show();
#
##
#print "------- using projected gradient method ---------";
#realCost = objTrue;
#C0 = np.multiply(STensor, x0) + ScTensor;
#startPop = 1.0* demand;
##if isFixed == False:
##    startPop = np.zeros(len(demand));
#V_pgm, pol_pgm, Cost_pgm, traj_pgm, error_pgm = pgm.projectGradient(STensor, ScTensor, D, Dc, C0, P, 8e-1, 1e-5, realCost,startPop, isFixed);
##
###objConstantExtra = 0.0;
###if isFixed:
###    objConstantExtra += demand.dot(V_pgm[:,0]);
###if isFixed == False: 
###    psiDemandInv = np.divide(demand - Dc, D);
###    objConstantExtra += -np.sum(np.divide(np.multiply(0.5* psiDemandInv - Dc,psiDemandInv ), D));
### ------PGM only-----------#
##plt.figure();  
##plt.plot(abs((error_pgm - realCost)/realCost),label = "running average difference");
##plt.title("Sub-gradient Method convergence to cvxpy optimal objective")
##plt.xlabel("Iteration");
##plt.ylabel(r'{(f(x_{ave}) - f^*)}/{f^*}')
##plt.yscale('log');
##plt.legend();
##plt.grid();
##plt.show();
###
##
##FWCost  = np.multiply(STensor, xStar) + ScTensor;
##np.max(FWCost - Cost_pgm)
#
####------------ graph of density propagation on time axis ------- #
##timeLine = np.linspace(1,Time,Time)
##plt.figure();
##for s in range(nodes):
##    plt.plot(timeLine,np.sum(xStar[s,:,:],axis = 0), label='state %d'%(s+1));
##
##plt.legend(fontsize = 'xx-small');
##plt.show();
###------------ graph of steadystate density on the actual graph ------- # 
##plt.figure();
##mag = 2.;
##cap = mag* np.ones(nodes); 
###nx.draw_networkx_nodes(ER,graphPos,node_size=cap,node_color='r',alpha=1);
##nx.draw(ER, pos=graphPos, node_size=200.*cap, node_color='w', edge_color='k',font_size = '20',with_labels=True, font_weight='bold');
##steadyState = np.einsum('ij->i', xStar[:,:,Time-1]);
##initialPopulation =  np.einsum('ij->i', xStar[:,:,0]);
##nodesize=[steadyState[f-1]*steadyState[f-1] for f in ER];
##original = [initialPopulation[f-1]*initialPopulation[f-1] for f in ER];
##nx.draw_networkx_nodes(ER,graphPos,node_size=original*cap,node_color='g',with_labels=True, alpha=0.5)
##nx.draw_networkx_nodes(ER,graphPos,node_size=nodesize*cap,node_color='r',with_labels=True, alpha=0.2)
##plt.show(); 
##------This is just graph----------# 
##plt.figure();
##
##nx.draw(ER, pos=graphPos, node_size=300.*cap, node_color='w', edge_color='k',font_color = 'k',font_size = '20',with_labels=True, font_weight='bold');
##nx.draw_networkx_nodes(ER,graphPos,node_size=350*cap,node_color='g',with_labels=True, alpha=0.3)
##
##plt.show();
##
###------------ histogram of playing vs quitting population ------- # 
##states = np.linspace(1,nodes,nodes)
##plt.figure();
##barWidth = 0.8;
##r1 = np.arange(len(original))
##r2 = [x + barWidth for x in r1]
##plt.bar(r1, demand, width = barWidth, edgecolor = 'black', label = 'quit game');
##plt.bar(r1, initialPopulation, width = barWidth, edgecolor = 'black', label= 'play game');
###plt.bar(r2, zStar, width = barWidth,  edgecolor = 'black', label = 'quit game');
##plt.xticks([r + barWidth for r in range(nodes)], (states));
##plt.legend();
##plt.show();
##
##
#plt.figure();
#plt.plot(abs((xK- objTrue)/objTrue),label = "Alg.7"); # Frank Wolfe Method
#plt.plot(abs((error_pgm - realCost)/realCost),label = "Alg.9"); # Projected Gradient Method
#plt.yscale("log");
#plt.grid();
#plt.legend();
#plt.show();
#
##
#plt.figure();
#plt.plot((xK),label = "Alg.7"); # Frank Wolfe Method
#plt.plot(( error_pgm),label = "Alg.9"); # Projected Gradient Method
#plt.plot(realCost*np.ones(500), label = 'optimal value', linestyle= '-.');
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
##plt.yscale("log");
#plt.grid();
#plt.legend();
#plt.show();
##    