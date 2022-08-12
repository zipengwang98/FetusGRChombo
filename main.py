from unittest import result
import initial_data
from flatten import *
import gauge_evolution
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
initial_data = initial_data.generate_initial_data()
flattened_initial_data = flatten(initial_data)

time_array= [0,1,2,3]
def zero_function(huge_flattened_array,t):
    return np.zeros(len(huge_flattened_array))

result_flat = odeint(zero_function,flattened_initial_data,time_array)
result = []
for i in range(len(result_flat)):
    result.append(unflatten(result_flat[i]))
result = np.array(result)

print(initial_data[17][1][2])

