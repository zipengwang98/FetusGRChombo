twothirds = 2.0/3.0

def dt_K(gammaUU_ij, D, alpha, tildeA_ij, K, rho, S, beta):
    '''
    Evaluates rhs of Eq. 11.52 to evolve K
    '''
    dt_K = -alpha*np.einsum('ij','j','i',gammaUU_ij,) + alpha*(einsum('ij','ij',tildeADD_ij,tildeAUU_ij))
    

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