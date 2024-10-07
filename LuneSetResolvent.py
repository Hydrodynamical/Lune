import numpy as np              # arrays
import matplotlib.pyplot as plt # plotting
from tqdm import tqdm           # loading bar
from concurrent.futures import ProcessPoolExecutor, as_completed


def is_in_lune_set(p, k, N):
    """Check if p \in R^2 is in the lune set 
    L(k, N) = \{p : |p| > N, |p - k| \leq N \}.
    """
    return (np.linalg.norm(p) > N) and (np.linalg.norm(p - k) <= N)

def generate_lune_set(k, N):
    """Generate the lune set as a list of np.arrays.
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

def count_lune_set(k,N):
    count = 0
    cutoff = N + np.max(np.abs(k))
    for i in range(-cutoff, cutoff + 1, 1):
        for j in range(-cutoff, cutoff + 1, 1):
            p = np.array([i,j])
            if is_in_lune_set(p, k, N):
                count+=1
    return count

def LuneResolvent(k, N):
    result = 0.0
    lune_set, _ = generate_lune_set(k, N)
    for p in lune_set:
        result += 1/(np.linalg.norm(p) ** 2 - np.linalg.norm(k - p) ** 2)
    return result

# print(LuneResolvent(k = np.array([5,7]), N = 1000))
# returns 1.5707564620188683


# Plot the function
k = np.array([5,7])

def calculate_resolvent(n):
    return LuneResolvent(k, N=n)

x = np.arange(100, 5000, 50)

# Use ProcessPoolExecutor for parallel computation
with ProcessPoolExecutor() as executor:
    # Submit tasks to the executor and wrap them in tqdm for progress tracking
    futures = [executor.submit(calculate_resolvent, n) for n in x]
    
    # Use as_completed to process the results as they finish
    results = []
    for future in tqdm(as_completed(futures), total=len(x), desc="Calculating LuneResolvent"):
        results.append(future.result())

plt.plot(x, results)
plt.xlabel('Fermi Radius N')
plt.ylabel('S(k, N)')
plt.show()
