from unittest import result
import initial_data
from flatten import *
import gauge_evolution
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from BSSN_rhs import *

initial_data = initial_data.generate_initial_data()
flattened_initial_data = flatten(initial_data)
dx = L / N

time_array= [0,1,2,3]
def zero_function(huge_flattened_array,t):
    return np.zeros(len(huge_flattened_array))

result_flat = odeint(BSSN_RHS,flattened_initial_data,time_array, args=(dx,))
result = []
for i in range(len(result_flat)):
    result.append(unflatten(result_flat[i]))
result = np.array(result)

print(initial_data[17][int(N/2)][int(N/2)])

