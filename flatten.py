import numpy as np
from initial_data import *

def flatten(data):
    return np.ndarray.flatten(data)
def unflatten(data):
    return np.reshape(data, (24,N,N,N))