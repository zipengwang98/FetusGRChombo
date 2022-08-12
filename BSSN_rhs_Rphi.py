import numpy as np
from tensor_algebra import * 

partial_betaU_trace = einsum('ii', partial(betaU))

def phi_rhs(alpha, betaU, phi, K, partial_betaU_trace):
    """ Calculate RHS of BSSN evolution equation for phi.
    """
    partial_phiD = partial(phi, dx)
    dtdphi = - 1/6 * alpha * K + einsum('i,i', betaU, deriv_phi) + 1/6 * partial_betaU_trace
    return dtdphi

def Rphi_ij(phi, barGammaDD, gammaUU, gammaDD):
    """
    phi part of traceless Ricci tensor decomposition
    """
    christoffelsymbolUDD = ChristoffelSymbolSecondKindDDD(gammaDD, dx)
    partial_phiD = partial(phi,dx)
    covphiDD = covariant_derivative_covector(partial(phi,dx), christoffelsymbolUDD)  
    return -(2.0 * covphiDD) - (2.0 * barGammaDD * np.einsum('ij','ji',gammaUU, covphiDD) + 4 * np.outer(partial_phiD, partial_phiD) -\
                                    (4 * barGamma * np.einsum('ij','j','i',gammaUU, partial_phiD, partial_phiD)    
