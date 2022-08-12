def RHS_alpha(alpha, K, beta_u1,beta_u2,beta_u3, d_l1_alpha, d_l2_alpha, d_l3_alpha):
    rhs = - 2 * alpha * K + (beta_u1 * d_l1_alpha + beta_u2 * d_l2_alpha + beta_u3 * d_l3_alpha)
    return rhs

def RHS_beta_ui(B_ui):
    rhs = 0.75 * B_ui
    return rhs

"""def RHS_beta_u2(B_u2):
    rhs = 0.75 * B_u2
    return rhs

def RHS_beta_u3(B_u3):
    rhs = 0.75 * B_u3
    return rhs"""

def RHS_B_ui(d_t_tildeGamma_ui, B_ui):
    eta=1 #constant of order 1/(2M)
    rhs = d_t_tildeGamma_ui - eta *  B_ui #d_t_tildeGamma_u1 is known via the evolution equations
    return rhs


"""def RHS_B_u2(d_t_tildeGamma_u2, B_u2):
    eta=1 #constant of order 1/(2M)
    rhs = d_t_tildeGamma_u1 - eta *  B_u1 #d_t_tildeGamma_u1 is known via the evolution equations
    return rhs


def RHS_B_u3(d_t_tildeGamma_u3, B_u3):
    eta=1 #constant of order 1/(2M)
    rhs = d_t_tildeGamma_u3 - eta *  B_u3 #d_t_tildeGamma_u1 is known via the evolution equations
    return rhs"""
