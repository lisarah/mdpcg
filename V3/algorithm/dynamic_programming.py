# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:18:30 2021

Dynamic programming methods: 
    - value iteration for both maximizing and minimizing objective.
    
@author: Sarah Li
"""
import numpy as np

def value_iteration_dict(cost, P, is_max=False):
    """ Value iteration with max/min objectives for a finite time horizon, 
        total cost MDP whose transition and costs are dictionaries
    
    Inputs:
        cost: list. [d_i] for i = 0...T-1
            d_i: dict. {(s,a): C_isa} for s in States, a in Actions
                C_isa: float. The cost of (s,a) at time i.
        
        P: list. [d_t] for i = 0...T-1
            d_t: dict. {s: P_ts} for s in States
                P_ts: dict. {a: P_tsa} for a in Actions 
                    P_tsa: (s_list, p_list).
                        s_list: List [s_j] s_j in [Neighbors of s]
                            s_j: tuple(int, int). (zone_ind, queue_level)
                        p_list: List [p_tsaj] 
                            P_tsaj: float. probability of transitioning 
                                into s_j from (t,s,a)
        is_max: bool. True if maximizing reward. False if minimizing cost.
    Returns:
        V: list. [V_t] for t in 0 ... T-1.
            V_t: dict. {s: V_ts} for s in States. 
                V_ts: float. The cost to go of state s at time t. 
        pol: list. [pol_t] for t in 0 ... T-1
            pol_t: dict. {s: pol_ts} for s in States.
                pol_ts: int in Actions. Optimal policy of state s at time t.   
    """
    T = len(P)
    V = []
    pol = []
    for rev_t in range(T):
        t = T - rev_t - 1
        V.append({s: -1 for s in P[t].keys()})
        pol.append({s: -1 for s in P[t].keys()})

        for s in P[t].keys():
            P_ts = P[t][s]
            if t == T-1:
                Q = {cost[t][(s,a)]:a  for a in P_ts.keys()}
            else:
                Q = {}
                for a in P[t][s].keys():
                    Q_sa = cost[t][(s,a)] + sum([
                        P_ts[a][1][i] * V[-2][P_ts[a][0][i]] 
                        for i in range(len(P_ts[a][0]))])
                    Q[Q_sa] = a
            V[-1][s] = max(Q.keys()) if is_max else min(Q.keys())
            pol[-1][s] = Q[V[-1][s]]  
            
    V.reverse()
    pol.reverse()
    return V, pol
 
def density_retrieval(pol, game):
    """ Given initial state distribution and a finite horizion dynamics and
        policy, determine the corresponding station-action density.
    
    Inputs:
        pol: list. [pol_i] for i = 0...T-1
            pol_i: dict. {s: a} for all s in States
                a = optimal policy to take in state s at time i.
        game: a queued_game object.
        
    Returns:
        sa_density: list. [d_i] for t= 0...T-1
            d_i: dict. {s: d_{0sa}} for s in States. 
            
    """
    T = len(pol)
    sa_density = []
    s_density = [game.t0]
    for t in range(T):
        # d_sum = sum([s_density[-1][s] for s in game.state_list])
        # assert round(d_sum, 5) == 100, f' density at time {t} is {d_sum}'
        sa_density.append({sa: 0  for sa in game.sa_list})
        for s in game.state_list:
            sa_density[-1][(s, pol[t][s])] = s_density[t][s]
        # sa_sum = sum([sum([sa_density[-1][(s,a)] for a in game.action_dict[s]]) 
        #               for s in game.state_list])
        # assert round(sa_sum, 5) == 100, f' density at time {t} is {sa_sum}'
        s_density.append(game.propagate(sa_density[t], t)) 
    
    return sa_density, s_density  

def value_iteration(cost, p0, P, isMax = False):
    """ Value iteration with max/min objectives for a finite time horizon, total
    cost MDP.
    
    Inputs:
        cost: np array with shape (S, A, T).
        p0: initial probabilitiy distribution, np array with length S
        P: transition kernel, np array with shape (S, S, A, T), P[s,r,a,t] is
          the probability of transition from state r to state s by taking 
          action a at time t. 
    Returns:
        V: values of each state, np array with length S.
        x_next: the optimal population distribution, np array with shape 
        (S,A,T).
    """
    S, A, T = cost.shape;
    V = np.zeros((S, T));
    policy = np.zeros((S, T)); # pi_t(state) = action;
    trajectory = np.zeros((S,T));
    xNext = np.zeros((S,A,T));

    # construct optimal value function and policy
    for tIter in range(T):
        t = T-1-tIter # true time since we are backward propagating.
        cCurrent =cost[:,:,t]; 
        if t == T-1:       
            if isMax:
                V[:,t] = np.max(cCurrent, axis = 1);
                policy[:,t] = np.argmax(cCurrent, axis=1);
            else:                 
                V[:,t] = np.min(cCurrent, axis = 1);
                policy[:,t] = np.argmin(cCurrent, axis=1);
        else:
            # solve the Bellman operator
            Vt = V[:,t+1];
            obj = cCurrent + np.einsum('ijk,i',P[t,:,:,:],Vt);
            if isMax:
                V[:,t] = np.max(obj, axis=1);
                policy[:,t] = np.argmax(obj, axis=1);
            else:
                V[:,t] = np.min(obj, axis=1);
                policy[:,t] = np.argmin(obj, axis=1);

    # construct the optimal trajectory corresponding to the Bellman operator.
    for t in range(T):
        if t == 0:
            traj = 1.0*p0;
        else:
            traj = trajectory[:,t-1];
        # construct y
        pol = policy[:,t];
        x = np.zeros((S,A));

        for s in range(S):
            x[s,int(pol[s])] = 1* traj[s];
        xNext[:,:,t] = 1.0*x;
        trajectory[:,t] =  np.einsum('ijk,jk',P[t,:,:,:],x);

    return V, xNext; 

