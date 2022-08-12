from math import gamma
import re
import numpy as np


def partial(f,n):

    df_n = f
    return(df_n)

def partial(f,dx):
    df_i = (np.roll(f,-1,axis=-3)-np.roll(f,1,axis=-3))/2/dx
    df_j = (np.roll(f,-1,axis=-2)-np.roll(f,1,axis=-2))/2/dx
    df_k = (np.roll(f,-1,axis=-1)-np.roll(f,1,axis=-1))/2/dx

    df = np.stack([df_i,df_j,df_k],axis=0)
    return(df)

def partial_vector(fU,dx):
    dfU_i = partial(fU[0],dx)
    dfU_j = partial(fU[1],dx)
    dfU_k = partial(fU[2],dx)
    dfU = np.stack([dfU_i,dfU_j,dfU_k],axis=1)
    return(dfU)

def partial_tensor(fUU,dx):
    dfUU_i = partial_vector(fUU[0],dx)
    dfUU_j = partial_vector(fUU[1],dx)
    dfUU_k = partial_vector(fUU[2],dx)
    dfUU = np.stack([dfUU_i,dfUU_j,dfUU_k],axis=1)
    return(dfUU)

def ChristoffelSymbolFirstKindDDD(gammaDD,dx):
    dgammaDDD = partial_tensor(gammaDD,dx)
    christoffelsymbolDDD = 0.5*(-dgammaDDD+np.transpose(dgammaDDD,(2,0,1,3,4,5))+np.transpose(dgammaDDD,(1,2,0,3,4,5)))
    return(christoffelsymbolDDD)

def raise_vector_index(fD,gammaDD):
    fU=fD
    return(fU)

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
    

#def 
#np.einsum('ij,ij',KUU,KDD)

