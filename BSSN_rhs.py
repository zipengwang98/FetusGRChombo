import numpy as np
from tensor_algebra import *
from flatten import *

twothirds = 2.0/3.0
onethird = 1.0/3

def dt_K(gammaDD ,gammaUU , alpha, tildeADD , K, betaU ):
    '''
    Evaluates rhs of Eq. 11.52 to evolve K
    '''
    christoffelsymbolUDD = ChristoffelSymbolSecondKindUDD(gammaDD ,dx)
    DDalphaDD  = covariant_derivative_covector(partial(alpha),christoffelsymbolUDD)
    tildeAUU  = raise_tensor_index(tildeADD , gammaUU )
    dt_K = -np.einsum('ij,ij',gammaUU ,DDalphaDD ) + alpha*(np.einsum('ij,ij',tildeADD ,tildeAUU )+onethird*K**2) + np.einsum("i,i",betaU ,partial(K))
    return (dt_K)

def dt_barGammaDD(N,alpha, tildeADD , betaU , barGammaDD ):
    '''
    Evaluates rhs of Eq. 11.51 to evolve barGammaDD
    '''
    dt_barGammaDD  = -2.0*alpha*tildeADD  + np.einsum('k,kij',betaU , partial_tensor(barGammaDD ,dx)) +\
                            np.einsum('ik,jk->ij',barGammaDD , partial_vector(betaU )) + np.einsum('kj,ik->ij',barGammaDD , partial_vector(betaU )) -\
                            twothirds*barGammaDD *np.einsum('kk',partial_vector(betaU ))
    return(dt_barGammaDD )

def dt_barChristofelU(tildaAUU,alpha,barChristofelUDD,barGammaUU,phi,barChristofelU,betaU ,K):
    d_beta_trace = np.einsum("jj",partial_vector(betaU ))
    dt_barChristofelU = -2*np.einsum("ij,j->i",tildaAUU*partial(alpha)) + 2*alpha*(np.einsum("ijk,kj->i",barChristofelUDD,tildaAUU)-\
        twothirds*("ij,j",barGammaUU,partial(K)) + 6*np.einsum("ij,j->i",tildaAUU,partial(phi))) +\
        np.einsum("j,ji->i",betaU ,partial_vector(barChristofelU))- np.einsum("j,ji->i",barChristofelU,partial_vector(betaU )) +\
        twothirds*barChristofelU*d_beta_trace + onethird*np.einsum("ji,j->i",barGammaUU,partial(d_beta_trace)) + np.einsum("lj,jli->i",barGammaUU,partial_tensor(partial_vector(betaU )))
    return(dt_barChristofelU)

def dt_phi(alpha, betaU, phi, K, partial_betaU_trace):
    """ Calculate RHS of BSSN evolution equation for phi.
    """
    partial_phiD = partial(phi, dx)
    dtdphi = - 1/6 * alpha * K + np.einsum('i,i', betaU, deriv_phi) + 1/6 * partial_betaU_trace
    return dtdphi

def BSSN_RHS(Huge_list, t):
    
    unflattenedlist = unflatten(Huge_list)
    alpha = unflattenedlist[17]
    print("alpha: ")
    print(np.max(alpha))

    out = np.zeros_like(Huge_list)
    return out
