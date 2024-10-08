import numpy as np              # arrays
import matplotlib.pyplot as plt # plotting
from tqdm import tqdm           # loading bar
import time                     # timestamp

def is_in_lune_set(p, k, N):
    """Check if p \in R^2 is in the lune set 
    L(k, N) = \{p : |p| > N, |p - k| \leq N \}.
    """
    return (np.linalg.norm(p) > N) and (np.linalg.norm(p - k) <= N)

def generate_lune_set(k, N):
    """Generate the lune set as a list of np.arrays and the count of points in the lune set.
    """
    cutoff = N + np.max(np.abs(k))
    lune_set = []
    count = 0
    for i in range(-cutoff, cutoff + 1, 1):
        for j in range(-cutoff, cutoff + 1, 1):
            p = np.array([i,j])
            if is_in_lune_set(p, k, N):
                lune_set.append(p)
                count += 1
    return lune_set, count

def LuneResolvent(k, N):
    result = 0.0
    lune_set, _ = generate_lune_set(k, N)
    for p in lune_set:
        result += 1/(np.linalg.norm(p) ** 2 - np.linalg.norm(k - p) ** 2)
    return result

# print(LuneResolvent(k = np.array([5,7]), N = 1000))
# returns 1.5707564620188683


current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
filename = f'data/LuneResolvent-{current_time}.npy'

# Save the function to a file
k = np.array([5,7])
x = np.arange(100, 200, 10)
y = [LuneResolvent(k, N=n) for n in tqdm(x, desc='Calculating LuneResolvent')]
data = np.column_stack((x, y))
np.save(filename, data)

