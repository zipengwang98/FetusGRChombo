import initial_data
import gauge_evolution
import numpy as np
from scipy.integrate import odeint
initial_data = initial_data.generate_initial_data()
flattened_initial_data = np.ndarray.flatten(initial_data)
time_array= [0,1,2,3]
def zero_function(huge_flattened_array,t):
    return np.zeros(len(huge_flattened_array))

result = odeint(zero_function,flattened_initial_data,time_array)



print(max(result[0]))
print(max(result[1]))
print(max(result[2]))
print(max(result[3]))

