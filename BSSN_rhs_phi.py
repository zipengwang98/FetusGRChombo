import numpy as np

partial_betaU_trace = einsum('ii', partial(betaU))

def phi_rhs(alpha, betaU, phi, K, partial_betaU_trace):
    """ Calculate RHS of BSSN evolution equation for phi.
    """
    deriv_phi = partial(phi)
    dtdphi = - 1/6 * alpha * K + einsum('i,i', betaU, deriv_phi) + 1/6 * partial_betaU_trace
    return dtdphi

def R_ij_phi(phi, tildegDD_ij):
    """
    Phi part of traceless Ricci tensor decomposition
    """
    covphiDD_ij =  
    covphiUD_kk =
    return -2 * covphiDD_ij - 2 * tildegDD_ij * covphiUD_kk + 4 * covaraint(phi) * covariant(phi) - 4 * tildegDD_ij * covariant(phi) * covariant(phi)   
