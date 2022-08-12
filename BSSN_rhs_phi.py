import numpy as np
from tensor_algebra import * 

partial_betaU_trace = einsum('ii', partial(betaU))

def phi_rhs(alpha, betaU, phi, K, partial_betaU_trace):
    """ Calculate RHS of BSSN evolution equation for phi.
    """
    deriv_phi = partial(phi)
    dtdphi = - 1/6 * alpha * K + einsum('i,i', betaU, deriv_phi) + 1/6 * partial_betaU_trace
    return dtdphi

def R_ij_phi(phi, bar_gamma_ij):
    """
    phi part of traceless Ricci tensor decomposition
    """
    partial_phi = partial(phi)
    covphiDD_ij =  
    covphiUD_kk =
    return -2 * covphiDD_ij - 2 * bar_gamma_ij * covphiUD_kk + 4 * covaraint(phi) * covariant(phi) - 4 * bar_gamma_ij * covariant(phi) * covariant(phi)   
