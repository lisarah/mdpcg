# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 12:09:47 2018

@author: sarah
These are helpers for mdpRouting game class
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class parameters:
    def __init__(self):
        self.tau = 27.  # $/hr
        self.vel = 8. # mph
        self.fuel = 28. # $/gal
        self.fuelEff = 20. # mi/gal
        self.rate = 6. # $/mi
        
sDEMAND = np.array([50 , 
                    100,
                    30 ,
                    120,
                    30 ,
                    80 ,
                    80 ,
                    20 ,
                    20 ,
                    70 ,
                    20 ,
                    20 ])  ;               
# start all drivers uniformly in residential neighbourhoods                                   
def resInit(nodes, residentialNum=0.1):
    # define initial condition -- same for both games
    p0 = np.zeros((nodes));
    
    # make all drivers start from residential areas 6 of them
    p0[2] = 1./residentialNum;
    p0[3] = 1./residentialNum;
    p0[7] = 1./residentialNum;
    p0[8] = 1./residentialNum;
    p0[10] = 1./residentialNum;
    p0[11] = 1./residentialNum;  
    return p0;
                                   
# ----------generate new constrained reward based on exact penalty-------------
# (for 2D reward matrix)
def constrainedReward2D(c,toll,constrainedState, time):
    states, actions = c.shape;
    constrainedC = np.zeros((states,actions,time));
    
    for t in range(time):
        constrainedC[:,:,t] = c;
        if toll[t] > 1e-8:
            for a in range(actions):
                constrainedC[constrainedState,a,t] += -toll[t];
    return constrainedC;
# (for 3D reward matrix):
def constrainedReward3D(c,toll,constrainedState):
    states, actions,time = c.shape;
    constrainedC = 1.0*c;

    for t in range(time):
        if toll[t] > 1e-8:
            for a in range(actions):
                constrainedC[constrainedState,a,t] += -toll[t];
                
    return constrainedC;   

def drawOptimalPopulation(time,pos,G,optRes,
                          constrainedState = None, 
                          startAtOne = False, 
                          constrainedUpperBound = 0.2,
                          numPlayers = 1.):
    frameNumber = time;
    v = G.number_of_nodes();
#    fig = plt.figure();
    #ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    #line, = ax.plot([], [], lw=2)
    iStart = -2;
    mag = numPlayers;
    cap = mag* np.ones(v);  
    if(constrainedState != None):
        # Draw the red circle
        cap[constrainedState]= cap[constrainedState]/(constrainedUpperBound)+1000; 

#    nx.draw_networkx_nodes(G,pos,node_size=cap,node_color='r',alpha=1);
    
    if(constrainedState != None):
        # Draw the white circle
        cap[constrainedState]= cap[constrainedState]/(constrainedUpperBound); 
        
    nx.draw(G, pos=pos, node_size=cap/mag, node_color='w',with_labels=True, font_weight='bold', font_color = 'w');
    dontStop = True; 
    try:
        print('running')
    except KeyboardInterrupt:
        print('paused')
        inp =input('continue? (y/n)')
    
    for i in range(iStart,frameNumber):
        try:
            if i < 0:
                frame = np.einsum('ij->i', optRes[:,:,0]);
            else:   
                frame = np.einsum('ij->i', optRes[:,:,i]);
                
            if startAtOne:
                nodesize=[mag*frame[f-1]*frame[f-1] for f in G];
            else:
                nodesize=[mag*frame[f]*frame[f] for f in G];
            if i  > 0:
                nx.draw_networkx_nodes(G,pos,node_size=lastNodeSize,node_color='w',alpha= 1);
            nx.draw_networkx_nodes(G,pos,node_size=nodesize,node_color='b',alpha=1, with_labels=True, font_weight='bold')  ;
            lastNodeSize = [];
            for i in nodesize:
                lastNodeSize.append(i+numPlayers*1.5);
        except KeyboardInterrupt:
            dontStop = False;
        plt.show();
        plt.pause(0.5);
       
def generateGridMDP(v,a,G,p = 0.8,test = False):
    """
    Generates a grid MDP based on given graph. p is the probability of reaching the target state given an action.
    
    Parameters
    ----------
    v : int
        Cardinality of state space.
    a : int
        Cardinality of input space.
        
    Returns
    -------
    P : (S,S,A) array
        Transition probability tensor such that ``P[i,j,k]=prob(x_next=i | x_now=j, u_now=k)``.
    """
    debug = False;
    # making the transition matrix
    P = np.zeros((v,v,a));
    for node in range(v):#x_now = node
        neighbours = list(G.neighbors(node));
        totalN = len(neighbours);
        # chance of not reaching action
        pNot = (1.-p)/(totalN-1);
        actionIter = 0;
        if debug: 
            print (neighbours);
        for neighbour in neighbours: # neighbour = x_next
            P[neighbour,node,actionIter] = p;
            for scattered in neighbours:
                if debug:
                    print (scattered);
                if scattered != neighbour:
                    P[scattered,node,actionIter] = pNot;
            actionIter += 1;
        while actionIter < a:         
            P[node, node, actionIter] = p;
            pNot = (1.-p)/(totalN);
            for scattered in neighbours: 
                P[scattered,node,actionIter] = pNot;
            actionIter += 1;
    # making the cost function
    if test:
        c = np.zeros((v,a));
        c[12] = 10.;
    else:
        c = np.random.uniform(size=(v,a))
    return P,c;



        
def generateMDP(v,a,G, p =0.9):
    """
    Generates a random MDP with finite sets X and U such that |X|=S and |U|=A.
    each action will take a state to one of its neighbours with p = 0.7
    rest of the neighbours will get p =0.3/(n-1) where n is the number of 
    neighbours of this state
    Parameters
    ----------
    S : int
        Cardinality of state space.
    A : int
        Cardinality of input space.
        
    Returns
    -------
    P : (S,S,A) array
        Transition probability tensor such that ``P[i,j,k]=prob(x_next=i | x_now=j, u_now=k)``.
    c : (S,A) array
        Cost such that ``c[i,j]=cost(x_now=i,u_now=j)``.
        
        
        
        COST IS Cy one^T - D
    """
    debug = False;
    P= np.zeros((v,v,a)); d = np.zeros((v,a))
    for node in range(v):#x_now = node
        nodeInd = node+1;
        neighbours = list(G.neighbors(nodeInd));
        totalN = len(neighbours);
        # chance of not reaching action
        pNot = (1.-p)/(totalN);
        actionIter = 0;
        if debug: 
            print (neighbours);
        for neighbour in neighbours: # neighbour = x_next
            neighbourInd = neighbour - 1;
            P[neighbourInd,node,actionIter] = p;
            # chance of ending somewhere else
            for scattered in neighbours:
                scatteredInd = scattered -1;
                if debug:
                    print (scattered);
                if scattered != neighbour:
                    # probablity of ending up at a neighbour
                    P[scatteredInd,node,actionIter] = pNot;
            # some probability of staying stationary
            P[node,node,actionIter] =pNot;
            actionIter += 1;        
        while actionIter < a:  # chances of staying still      
            P[node, node, actionIter] = 1.0;
#            P[node, node, actionIter] = p;
#            pNot = (1.-p)/(totalN);
#            for scattered in neighbours: 
#                scatteredInd = scattered -1;
#                P[scatteredInd,node,actionIter] = pNot;
            actionIter += 1;
    # test the cost function
    c = 1000.*np.ones((v,a))
    c[6] = 0.;

    return P,c
def getDistance(node1, node2, dictionary):
    if node1 == node2:
        return 0.0;
    elif node1 < node2:
        a = node1; b = node2;
    else: 
        a = node2; b = node1;
    return dictionary[(a,b)]; 
def getExpectedDistance(node, G, dictionary):
    neighbours = list(G.neighbors(node));
    totalDistance = 0.;
    for neighbour in neighbours:
        totalDistance += getDistance(node, neighbour, dictionary);
    return totalDistance/len(neighbours);
        
def generateQuadMDP(v,a,G,distances, p =0.9):
    """
    Generates a random MDP with finite sets X and U such that |X|=S and |U|=A.
    each action will take a state to one of its neighbours with p = 0.7
    rest of the neighbours will get p =0.3/(n-1) where n is the number of 
    neighbours of this state
    Parameters
    ----------
    S : int
        Cardinality of state space.
    A : int
        Cardinality of input space.
        
    Returns
    -------
    P : (S,S,A) array
        Transition probability tensor such that ``P[i,j,k]=prob(x_next=i | x_now=j, u_now=k)``.
    c : (S,A) array
        Cost such that ``c[i,j]=cost(x_now=i,u_now=j)``.
        
        
        
        COST IS Cy one^T - D
    """
    
    
    debug = False;
    P= np.zeros((v,v,a)); c = np.zeros((v,a)); d = np.zeros((v,a))
    sP = parameters();
    # reward constant for going somewhere else
    kGo = sP.tau/sP.vel + sP.fuel/sP.fuelEff;
    for node in range(v):#x_now = node
        nodeInd = node+1;
        neighbours = list(G.neighbors(nodeInd));
        totalN = len(neighbours);
        evenP = 1./(totalN +1); # even probability of ending up somewhere when picking up
        # chance of not reaching action
        pNot = (1.-p)/(totalN);
        actionIter = 0;
        if debug: 
            print (neighbours);
        for neighbour in neighbours: # neighbour = x_next
            # ACTION = going somewhere else
            neighbourInd = neighbour - 1;
            P[neighbourInd,node,actionIter] = p;
            c[node, actionIter] = -kGo*getDistance(neighbour,nodeInd, distances);
            d[node, actionIter] = 0; # indedpendent of distance
            # chance of ending somewhere else
            for scattered in neighbours:
                scatteredInd = scattered -1;
                if debug:
                    print (scattered);
                if scattered != neighbour:
                    # probablity of ending up at a neighbour
                    P[scatteredInd,node,actionIter] = pNot;
            # some probability of staying stationary
            P[node,node,actionIter] =pNot;
            actionIter += 1;        
        while actionIter < a:  
            # ACTION  = picking up rider
            P[node, node, actionIter] = evenP;
            for scattered in neighbours: 
                scatteredInd = scattered -1;
                P[scatteredInd,node,actionIter] = evenP;
                
            c[node, actionIter] = (sP.rate - kGo)*getExpectedDistance(nodeInd,G,distances); # constant offset 
            d[node,actionIter] = sP.tau/sDEMAND[node]; # dependence on current density
#            P[node, node, actionIter] = p;
#            pNot = (1.-p)/(totalN);

            actionIter += 1;
    # test the cost function
#    c = 1000.*np.ones((v,a))
#    c[6] = 0.;

    return P,c,d     


def airportMDP():
    """
        Builds (S, A, P, C, D) of seattle airport
        S: Two types of states
            - S_i: waitline for concourse i
            - G_i: gate i in a concourse/satellite
        Seattle airport's gates: total = 77
            concourse A: 14 gates, s = 0
            concourse B: 12 gates, s = 1
            concourse C: 12 gates, s = 2
            concourse D: 11 gates, s = 3
            satellite N: 14 gates, s = 4
            satellite S: 14 gates, s = 5
        A: from different states we have different actions
            - S_i: 
                - A_i waits in S_i. goes to take off state with 0.5 probability
                  and 0.5 chance of staying in S_i
                - A_j goes to neighbouring states j with 0.9 probability 
                  and 0.1/N chance of other states

            - G_i: 
                - A_i stays in current state with probability 1
        P: Probability kernel, see actions 
        C: cost of each state action
            - (S_i, A_j): Gas money
            - (S_i, A_i): population dependent l(y) = Cy + d
            - (G_i, A_i): 1
            ** each (S_i, A_i/j) cost must be much higher than _i , A_i
        returns: 
            P: probability kernel, S x S x A
            C: cost of game l(y) = Cy + D, S x A
            D: cost of game l(y) = Cy + D, S x A
    """
    # States  -------------------------------------------
    # order of states goes: G_1, G_2, ... T_1, T_2,...
    # gate ordered from concourse A/B/C/D satellite N/S 
    #          -------------------------------------------
    S = 6 + 77;
    
    # Actions  -------------------------------------------
    # actions: going within each concourse/satellite, 
    #          going to neighbouring concourse/satellite
    # action order: first action is to stay in concourse
    #               subsequent actions are to move to neighbouring state
    #          -------------------------------------------
    A = np.zeros(S);
    # concourse A
    A[0] = 3;
    # concourse B
    A[1] = 4;
    # concourse C
    A[2] = 4;
    # concourse D
    A[3] = 3;
    # satellite N
    A[4] = 3;
    # satellite S
    A[5] = 3;
    # the gate states T_i
    A[6:] = 1;
#    print (A)
    actionSize = int( max(A));
    # corresponding terminal states
    Terminals = np.array([14,12,12, 11,14,14]);
    # neighbouring states 
    Neighbours = [[1,5],   # A is connected to (B, S)
                  [0,2,5], # B is connected to (A, C, S)
                  [1,3,4], # C is connected to (B, D, N),
                  [2,4],   # D is connected to (C, N)
                  [2,3],   # N is connected to (C, D)
                  [0,1]];  # S is connected to (A, B)
                  
    # probability kernel -------------------------------------------
    P = np.zeros((S,S,actionSize));
    for s in range(S):
        if s <= 5: # state is a concourse state
            # action 0: state in concourse state
            P[s,s,0] = 0.5;
            termStart = int(6 + sum(Terminals[0:s]));
            termEnd = termStart + Terminals[s];
            P[termStart:termEnd, s, 0] = 0.5/Terminals[s];
#            print ("state ", s );
#            print (P[:,s,0])
            # other actions: going to neighbouring concourses
            neighbourAction = 1;
            neighbours = len(Neighbours[s]) - 1;
            for desti in Neighbours[s]:
                P[desti, s, neighbourAction] = 0.9;
                for otherNeighB in Neighbours[s]:
                    if otherNeighB != desti:
                        P[otherNeighB, s, neighbourAction] = 0.1/neighbours;
                neighbourAction += 1;
        else: # state is a gate state
           P[s,s,0] = 1;
   
    # cost     -------------------------------------------
    # l(y) = Cy + D, C: S x A, D : S x A
    C = np.zeros((S,actionSize));
    D = np.zeros((S,actionSize));
    # cost of non-existent actions is infinity         
    for s in range(S):
        action = 0;
        # cost of feasible action
        while action < A[s]:
            if action == 0:
                if s <= 5: # cost of staying in concourse s
                    C[s, action]= 50;
                else: # cost of staying in gate s
                    C[s, action] = 0;
                D[s, action] = 1;
            else:
                # going to neighbouring states depends on how 
                # many other planes are traversing the road
                C[s, action] = 1; 
                # gas money of going else where
                D[s, action] = 3;
            action += 1;
        # cost of non-existent actions is infinity
        someInf = 10000; # apparently np.inf doesn't work
        while action < actionSize:        
            C[s, action] = 0;
            D[s, action] = someInf;
            action +=1;    
    return P, C, D, S, actionSize