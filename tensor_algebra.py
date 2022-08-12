from math import gamma
import re
import numpy as np

def partial(f,n):

    df_n = f
    return(df_n)

def raise_index(fUD,gamma):
    fUU=fUD
    return(fUU)

def lower_index(fUU,gamma):
    fUD=fUU
    return(fUD)

def physical_metric(bar_gamma, phi):
    '''give me phi, and bar_gamma, outputs the physical metric'''
    gamma_DD_ij = np.zeros(3,3)
    for i in [0,1,2]:
        for j in [0,1,2]:
            gamma_DD_ij[i][j] = bar_gamma[i][j] * exp(4 * phi)
    
    return gamma_DD_ij

def trace(A_DD, gamma_DD_ij):
    '''give my phi and bar_gamma, output the trace of a tensor A'''
    gamma_UU = np.linalg.inv(gamma_DD_ij)
    out = 0
    for i in [0,1,2]:
        for j in [0,1,2]:
                out[i][j] += gamma_UU[i][j] * A_DD[i][j]
    return out

def TF_of_a_tensor(A, phi, bar_gamma):
    '''give me a tensor A (DD), outputs its tracefree part'''
    gamma_DD_ij = physical_metric(bar_gamma, phi)
    A_trace = trace(A, gamma_DD_ij)
    out = np.zeros(3,3)
    for i in [0,1,2]:
        for j in [0,1,2]:
            out[i][j] = A[i][j] - 1/3.0 * gamma_DD_ij[i][j] * A_trace
    return out
    