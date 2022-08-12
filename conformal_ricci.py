import numpy as np
from tensor_algebra import *

def conformal_ricci():
    bar_ricci = - 0.5 * np.einsum('lm ,lmij->ij', barGammaUU, partial_tensor(partial_tensor(barGammaDD, dx), dx)) + \
                0.5 * (np.einsum('ki, jk->ij', barGammaDD, partial_vector(barChristoffelU, dx) + np.einsum('kj,ik->ij', barGammaDD, partial_ vector(barChristoffelU, dx))) + \
                0.5 * (einsum('k,ijk->ij', barChristoffelU, ChristoffelSymbolFirstKindDDD) + einsum('k,jik->ij', barChristoffelU, ChristoffelSymbolFirstKinDDD)) + \
                0.5 * ('lm, kli, jkm->ij', barGammaUU, ChristoffelSymbolSecondKindUDD, ChristoffelSymbolFirstKindDDD) + ('lm, klj, ikm->ij', barGammaUU, \ 
                    ChristoffelSymbolSecondKindUDD, ChristoffelSymbolFirstKindDDD) + \
                0.5 * (einsum('ki,jk->ij', barGammaDD, partial_vector(barChristoffelU, dx)) + einsum('kj,ik->ij', barGammaDD, partial_vector(barChristoffelU, dx)))
    return bar_ricci
