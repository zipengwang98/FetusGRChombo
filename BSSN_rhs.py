import numpy as np
from tensor_algebra import *
from flatten import *

twothirds = 2.0/3.0
onethird = 1.0/3
onesixth = 1./6

def barRDD(barGammaUU,barGammaDD,barChristoffelU, barChristoffelDDD,barChristoffelUDD):
    bar_ricci = - 0.5 * np.einsum('lm ,lmij->ij', barGammaUU, partial_tensor(partial_tensor(barGammaDD, dx), dx)) + \
                0.5 * (np.einsum('ki, jk->ij', barGammaDD, partial_vector(barChristoffelU, dx) + np.einsum('kj,ik->ij', barGammaDD, partial_vector(barChristoffelU, dx)))) + \
                0.5 * (np.einsum('k,ijk->ij', barChristoffelU,  barChristoffelDDD) + np.einsum('k,jik->ij', barChristoffelU, barChristoffelDDD)) + \
                0.5 * np.einsum('lm, kli, jkm->ij', barGammaUU, barChristoffelUDD,  barChristoffelDDD) + np.einsum('lm, klj, ikm->ij', barGammaUU,barChristoffelUDD, barChristoffelDDD) + \
                0.5 * (np.einsum('ki,jk->ij', barGammaDD, partial_vector(barChristoffelU, dx)) + np.einsum('kj,ik->ij', barGammaDD, partial_vector(barChristoffelU, dx)))
    return bar_ricci


def Rphi_ij(phi, barGammaDD, gammaUU, gammaDD):
    """
    Phi part of traceless Ricci tensor decomposition
    """
    christoffelsymbolUDD = ChristoffelSymbolSecondKindUDD(gammaDD,dx)
    partial_phiD = partial(phi,dx)
    covphiDD = covariant_derivative_covector(partial(phi,dx), christoffelsymbolUDD)
    Rphi = -(2.0 * covphiDD) - 2.0 * barGammaDD * np.einsum('ij, ji',gammaUU, covphiDD) + 4 * np.outer(partial_phiD, partial_phiD) -\
                                    4 * barGammaDD * np.einsum('ij','j','i',gammaUU, partial_phiD, partial_phiD)
    return Rphi

def RTF_ij(phi, barGammaDD, barGammaUU, gammaUU, gammaDD,barChristoffelU, barChristoffelDDD,barChristoffelUDD):
    RicciDD = Rphi_ij(phi, barGammaDD, gammaUU, gammaDD) + barRDD(barGammaUU,barGammaDD,barChristoffelU, barChristoffelDDD,barChristoffelUDD)
    RTFDD = TF_of_a_tensor(RicciDD, phi, barGammaDD)
    return RTFDD
    
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

def dt_barChristoffelU(tildaAUU,alpha,barChristoffelUDD,barGammaUU,phi,barChristoffelU,betaU ,K):
    d_beta_trace = np.einsum("jj",partial_vector(betaU ))
    dt_barChristoffelU = -2*np.einsum("ij,j->i",tildaAUU*partial(alpha)) + 2*alpha*(np.einsum("ijk,kj->i",barChristoffelUDD,tildaAUU)-\
        twothirds*("ij,j",barGammaUU,partial(K)) + 6*np.einsum("ij,j->i",tildaAUU,partial(phi))) +\
        np.einsum("j,ji->i",betaU ,partial_vector(barChristoffelU))- np.einsum("j,ji->i",barChristoffelU,partial_vector(betaU )) +\
        twothirds*barChristoffelU*d_beta_trace + onethird*np.einsum("ji,j->i",barGammaUU,partial(d_beta_trace)) + np.einsum("lj,jli->i",barGammaUU,partial_tensor(partial_vector(betaU )))
    return(dt_barChristoffelU)

def dt_phi(alpha, betaU, phi, K, dx):
    ''' Calculate RHS of BSSN evolution equation for phi.
    '''
    dt_dphi = - onesixth * alpha * K + np.einsum('i,i', betaU, partial(phi, dx)) + onesixth * np.einsum("ii",partial_vector(betaU,dx))
    return(dt_dphi)

def dt_tildaADD(phi,alpha,barGammaDD,K,tildaADD,gammaUU,betaU,barChristoffelU,gammaDD,barGammaUU,barChristoffelDDD,barChristoffelUDD):
    dt_tildaADD = np.exp(-4*phi)*(-TF_of_a_tensor(covariant_derivative_covector(partial(alpha)),phi, barGammaDD)+\
        alpha* RTF_ij(phi, barGammaDD, barGammaUU, gammaUU, gammaDD,barChristoffelU, barChristoffelDDD,barChristoffelUDD)) +\
         alpha*(K*tildaADD-2*np.einsum("il,lj->ij",tildaADD,raise_vector_index(tildaADD, gammaUU))) +\
        np.einsum("k,kij->ij",betaU,partial_tensor(tildaADD))+ np.einsum("ik,jk->ij",tildaADD,partial_vector(betaU)) - twothirds*tildaADD*np.einsum("kk",partial(betaU))
    return(dt_tildaADD)

def assemble_tensor_sym(components):
    ''' given a list of length 6, assemble the sym tensor'''
    a11 = components[0]
    a12 = components[1]
    a13 = components[2]
    a22 = components[3]
    a23 = components[4]
    a33 = components[5]
    out = np.array([[a11,a12,a13],
                     [a12, a22, a23],
                     [a13, a23, a33]])
    return out

def BSSN_RHS(Huge_list, t, dx):
    
    data_input = unflatten(Huge_list)
    phi = data_input[0]
    K = data_input[1]
    GammaChrist_vec_U = np.array([data_input[2],data_input[3],data_input[4]])

    bar_gamma_DD = assemble_tensor_sym(data_input[5:11])
    bar_A_DD = assemble_tensor_sym(data_input[11:17])
    alpha = data_input[17]
    beta_vec_U = np.array([data_input[18],data_input[19],data_input[20]])
    B_vec_U = np.array([data_input[21],data_input[22],data_input[23]])

    bar_gamma_UU = inverse_of_lists_of_tensor(bar_gamma_DD)

    out = np.zeros_like(Huge_list)

    phi_rhs = dt_phi(alpha, beta_vec_U,phi, K, dx)
    K_rhs = dt_K()
    return out
