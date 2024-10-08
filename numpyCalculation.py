import numpy as np
import time 
from tqdm import tqdm

def generate_lune_set(k, N):
    """Generate the lune set as an array of np.arrays."""
    cutoff = int(N + np.max(np.abs(k)))
    i = np.arange(-cutoff, cutoff + 1)
    j = np.arange(-cutoff, cutoff + 1)
    I, J = np.meshgrid(i, j, indexing='ij')
    P = np.stack((I, J), axis=-1)  # Shape: (len(i), len(j), 2)
    
    # Compute squared norms to avoid square roots
    norm_p2 = np.sum(P ** 2, axis=-1)
    norm_p_minus_k2 = np.sum((P - k) ** 2, axis=-1)
    
    # Boolean mask for points in the lune set
    mask = (norm_p2 > N ** 2) & (norm_p_minus_k2 <= N ** 2)
    lune_set = P[mask]
    return lune_set

def LuneResolvent(k, N):
    lune_set = generate_lune_set(k, N)
    
    # Simplify the denominator using algebra
    denom = 2 * np.dot(lune_set, k) - np.dot(k, k)
    
    # Avoid division by zero
    denom[denom == 0] = np.finfo(float).eps
    result = np.sum(1 / denom)
    return result

current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
filename = f'data/LuneResolvent-{current_time}.npy'

# Save the function to a file
k = np.array([5,7])
x = np.arange(5_000, 10_000, 100)
y = [LuneResolvent(k, N=n) for n in tqdm(x, desc='Calculating LuneResolvent')]
data = np.column_stack((x, y))
np.save(filename, data)
