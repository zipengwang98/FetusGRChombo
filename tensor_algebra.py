import numpy as np

def partial(f,dx):
    df_i = (np.roll(f,-1,axis=0)-np.roll(f,1,axis=0))/2/dx
    df_j = (np.roll(f,-1,axis=1)-np.roll(f,1,axis=1))/2/dx
    df_k = (np.roll(f,-1,axis=2)-np.roll(f,1,axis=2))/2/dx

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
    christoffelsymbolDDD = 0.5*(-dgammaDDD+np.transpose(dgammaDDD,(2,0,1,3))+np.transpose(dgammaDDD,(1,2,0,3)))
    return(christoffelsymbolDDD)

def raise_vector_index(fD,gammaDD):
    fU=fD
    return(fU)

def lower_index(fUU,gamma):
    fUD=fUU
    return(fUD)

#def 
#np.einsum('ij,ij',KUU,KDD)