# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:15:52 2021

Plotting purposes

@author: Sarah Li
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def latex_format():
    """ Format matplotlib output to be latex compatible with gigantic fonts."""
    mpl.rc('font',**{'family':'serif'})
    mpl.rc('text', usetex=True)
    mpl.rcParams.update({'font.size': 20})
    mpl.rc('legend', fontsize='small')

def objective(hist, optimal_value = None, alg_name='algorithm'):
    """ Plot the objective convergence behaviour against true optimal value."""
    blue = '#1f77b4ff';
    orange = '#ff7f0eff';
    plt.figure()
    if optimal_value is None:
        plt.plot(np.linspace(1, len(hist),len(hist)), 
                 [abs(x) for x in hist], 
                 linewidth=2, 
                 label=f'{alg_name}',
                 color=blue)
    else:
        plt.plot(np.linspace(1, len(hist),len(hist)), 
             [((x - optimal_value)/optimal_value) for x in hist], 
             linewidth=2, 
             label=f'{alg_name}',
             color=blue)
    plt.legend();
    plt.xlabel(r"Iterations")
    # plt.ylabel(r"$f(y^k)$")
    # plt.yscale("log")
    plt.xscale('log')
    plt.grid();
    plt.show();
