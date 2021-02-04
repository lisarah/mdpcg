# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:03:46 2021

CVX variable to value array conversions.

@author: Sarah Li
"""
import numpy as np

#----------convert cvx variable dictionary into an array of dictated shape
def cvxDict2Arr(optDict, shapeList):
    arr = np.zeros(shapeList);
    for DIter, key in enumerate(optDict):
        arr[key] = optDict[key].value;
    return arr
#----------convert cvx variable list into an array of dictated shape,
# mostly used for dual variables, since the cvx constraints are in lists
def cvxList2Arr(optList,shapeList,isDual):
    arr = np.zeros(shapeList);
    it = np.nditer(arr, flags=['f_index'], op_flags=['writeonly'])    
    for pos, item in enumerate(optList):
        print (item)
        if isDual:
            it[0] = item.dual_value;
        else:
            it[0] = item.value;
        
        it.iternext();                    
    return arr;
#----------convert cvx variable array into a value array
def cvx_array_2_array(variable_array):
    M, N = variable_array.shape
    value_array = np.zeros((M,N))  
    for i in range(M):
        for j in range(N):
            value_array[i,j] = variable_array[i,j].value             
    return value_array