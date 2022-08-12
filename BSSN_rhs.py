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

def barGamma(N,alpha, tildeADD_ij, betaU_i, barGammaDD_ij, ):
    '''
    Evaluates rhs of Eq. 11.51 to evolve barGamma
    '''
    dt_barGammaDD_ij = -2.0*alpha*tildeADD_ij + np.einsum('k','kij',betaU_i, partial()) +\
                            np.einsum('ik','jk',barGammaDD_ij, partial()) + np.einsum('kj','ik',barGammaDD_ij, partial()) -\
                            twothirds*barGammaDD_ij*partial()
    return dt_barGammaDD_ij

def BSSN_RHS(Huge_list, t):
    out = np.zeros_like(Huge_list)
    return out