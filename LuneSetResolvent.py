import numpy as np              # arrays
import matplotlib.pyplot as plt # plotting
from tqdm import tqdm           # loading bar
import concurrent.futures       # parallel computing

def is_in_lune_set(p, k, N):
    """Check if p \in R^2 is in the lune set 
    L(k, N) = \{p : |p| > N, |p - k| \leq N \}.
    
    Inputs:
        p: np.array
        k: np.array
        N: int
        
    Returns:
        bool
    """
    if (np.linalg.norm(p) > N) and (np.linalg.norm(p - k) <= N):
        return True
    else:
        return False
    
# # Test function is_in_lune_set
# N = 10
# test_p = np.array([3,4])
# test_k = np.array([-10,-1])
# print(is_in_lune_set(test_p, test_k))

def generate_lune_set(k, N):
    """Generate the lune set as a list of np.arrays.
    """
    cutoff = N + np.max(np.abs(k))
    lune_set = []
    for i in range(-cutoff, cutoff + 1, 1):
        for j in range(-cutoff, cutoff + 1, 1):
            p = np.array([i,j])
            if is_in_lune_set(p, k, N):
                lune_set.append(p)
    return lune_set

# # Test generate lune set
# N = 10
# test_k = np.array([-10,-1])
# print(generate_lune_set(test_k, N))

def LuneResolvent(k, N):
    result = 0.0
    lune_set = generate_lune_set(k, N)
    for p in lune_set:
        result += 1/(np.linalg.norm(p) ** 2 - np.linalg.norm(k - p) ** 2)
    return result

# print(LuneResolvent(k = np.array([5,7]), N = 1000))
# returns 1.5707564620188683


k=np.array([5, 7])
lune_set_list = [generate_lune_set(k, N = n) for n in range(100, 500, 10)]

# Parallelizing the computation
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(LuneResolvent, lune_set_list))

print(results)


# Plot the function
x = np.arange(1000, 5000, 100)
y = [LuneResolvent(k, N=n) for n in tqdm(x, desc="Calculating LuneResolvent")]
plt.plot(x, y)
plt.xlabel('Fermi Radius N')
plt.ylabel('S(k, N)')
plt.show()
