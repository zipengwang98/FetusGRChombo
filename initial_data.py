from params import *
import numpy as np

def generate_initial_data():
    dx = N/L
    det_initial_gamma = 1
    phi = (1/12)*np.log(det_initial_gamma/eta)

    zero_grid = np.zeros((N,N,N))
    one_grid = np.ones((N,N,N))
    
    initial_alpha_grid = np.copy(zero_grid)
    initial_phi_grid = np.copy(zero_grid)
    initial_beta_u1_grid = np.copy(zero_grid)
    initial_beta_u2_grid = np.copy(zero_grid)
    initial_beta_u3_grid = np.copy(zero_grid)
    initial_B_u1_grid = np.copy(zero_grid)
    initial_B_u2_grid = np.copy(zero_grid)
    initial_B_u3_grid = np.copy(zero_grid)
    initial_barGamma_u1_grid = np.copy(zero_grid)
    initial_barGamma_u2_grid = np.copy(zero_grid)
    initial_barGamma_u3_grid = np.copy(zero_grid)
    initial_K_grid = np.copy(zero_grid)
    initial_A_d1d1_grid = np.copy(zero_grid)
    initial_A_d1d2_grid = np.copy(zero_grid)
    initial_A_d1d3_grid = np.copy(zero_grid)
    initial_A_d2d2_grid = np.copy(zero_grid)
    initial_A_d2d3_grid = np.copy(zero_grid)
    initial_A_d3d3_grid = np.copy(zero_grid)
    initial_bargamma_d1d1_grid = np.exp(-4* phi)* np.copy(one_grid)
    initial_bargamma_d1d2_grid = np.exp(-4* phi)*np.copy(zero_grid)
    initial_bargamma_d1d3_grid = np.exp(-4* phi)*np.copy(zero_grid)
    initial_bargamma_d2d2_grid = np.exp(-4* phi)*np.copy(one_grid)
    initial_bargamma_d2d3_grid = np.exp(-4* phi)*np.copy(zero_grid)
    initial_bargamma_d3d3_grid = np.exp(-4* phi)*np.copy(one_grid)

    for x in range(N):
        for y in range(N):
            for z in range(N):
                initial_alpha_grid[x][y][z] =  np.exp(-(N**(-2))*(((x-N/2)*dx)**2+((y-N/2)*dx)**2+((y-N/2)*dx)**2)) 
                
    print(initial_alpha_grid)
    return np.array([
        initial_phi_grid,

        initial_K_grid,

        initial_barGamma_u1_grid,
        initial_barGamma_u2_grid,
        initial_barGamma_u3_grid,

        initial_bargamma_d1d1_grid,
        initial_bargamma_d1d2_grid,
        initial_bargamma_d1d3_grid,
        initial_bargamma_d2d2_grid,
        initial_bargamma_d2d3_grid,
        initial_bargamma_d3d3_grid,

        initial_A_d1d1_grid,
        initial_A_d1d2_grid,
        initial_A_d1d3_grid,
        initial_A_d2d2_grid,
        initial_A_d2d3_grid,
        initial_A_d3d3_grid,
        
        initial_alpha_grid,

        initial_beta_u1_grid,
        initial_beta_u2_grid,
        initial_beta_u3_grid,

        initial_B_u1_grid, 
        initial_B_u2_grid, 
        initial_B_u3_grid])
     

