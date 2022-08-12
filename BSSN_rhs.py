import numpy as np
from tensor_algebra import *

twothirds = 2.0/3.0
onethird = 1.0/3

def dt_K(gammaDD_ij,gammaUU_ij, D, alpha, tildeADD_ij, K, rho, S, betaU_i):
    '''
    Evaluates rhs of Eq. 11.52 to evolve K
    '''
    christoffelsymbolUDD = ChristoffelSymbolSecondKindUDD(gammaDD_ij,dx)
    DDalphaDD_ij = covariant_derivative_covector(partial(alpha),christoffelsymbolUDD)
    tildeAUU_ij = raise_tensor_index(tildeADD_ij, gammaUU_ij)
    dt_K = -np.einsum('ij,ij',gammaUU_ij,DDalphaDD_ij) + alpha*(np.einsum('ij,ij',tildeADD_ij,tildeAUU_ij)+onethird*K**2) + np.einsum("i,i",betaU_i,partial(K))
    return (dt_K)

def dt_barGammaDD(N,alpha, tildeADD_ij, betaU_i, barGammaDD_ij, ):
    '''
    Evaluates rhs of Eq. 11.51 to evolve barGammaDD
    '''
    dt_barGammaDD_ij = -2.0*alpha*tildeADD_ij + np.einsum('k,kij',betaU_i, partial_tensor(barGammaDD_ij,dx)) +\
                            np.einsum('ik,jk->ij',barGammaDD_ij, partial_vector(betaU_i)) + np.einsum('kj,ik->ij',barGammaDD_ij, partial_vector(betaU_i)) -\
                            twothirds*barGammaDD_ij*np.einsum('kk',partial_vector(betaU_i))
    return(dt_barGammaDD_ij)

def dt_barChristofelU(tildaAUU,alpha,barChristofelUDD,barGammaUU,phi,barChristofelU,betaU_i,K):
    dt_barChristofelU = -2*np.einsum("ij,j->i",tildaAUU*partial(alpha)) + 2*alpha*(np.einsum("ijk,kj->i",barChristofelUDD,tildaAUU)-\
        twothirds*("ij,j",barGammaUU,partial(K)) + 6*np.einsum("ij,j->i",tildaAUU,partial(phi))) +\
        betaU_i
    return(dt_barChristofelU)

def BSSN_RHS(Huge_list, t):
    out = np.zeros_like(Huge_list)
    return out
