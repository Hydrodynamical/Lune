import numpy as np              # arrays
import matplotlib.pyplot as plt # plotting
from tqdm import tqdm           # loading bar
import time                     # timestamp

# number theory stuff
from sympy import primerange, isprime
from math import gcd


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

def Resolvent(k, N):
    result = 0.0
    lune_set, _ = generate_lune_set(k, N)
    for p in lune_set:
        result += 1/(np.linalg.norm(p) ** 2 - np.linalg.norm(k - p) ** 2)
    return result

def generate_coprime_pairs(limit):
    coprime_pairs = []
    for a in range(1, limit):
        for b in range(a + 1, limit):
            if gcd(a, b) == 1:
                coprime_pairs.append((a, b))
    return coprime_pairs

def generate_prime_pairs(limit):
    primes = list(primerange(1, limit))
    prime_pairs = [(p1, p2) for i, p1 in enumerate(primes) for p2 in primes[i+1:]]
    return prime_pairs

def prime_coprime_pair(limit = 20):
    # Generate coprime and prime pairs
    coprime_pairs = generate_coprime_pairs(limit)
    prime_pairs = generate_prime_pairs(limit)
    return prime_pairs, coprime_pairs

def ResolventGraph(k, N):
    x = np.arange(100, N, 1)
    y = [Resolvent(k, n) for n in tqdm(x, desc='Calculating Resolvent')]
    return x, y

# print(LuneResolvent(k = np.array([5,7]), N = 1000))
# returns 1.5707564620188683

# print(prime_coprime_pair(20))

# results = []
# x = np.arange(100, 500, 10)
# for k in tqdm(generate_coprime_pairs(20), desc='Calculating Resolvent for coprime and prime pairs'):
#     k_array = np.array(k)
#     print(k_array)
#     y = [Resolvent(k_array, N=n) for n in x]
#     results.append((k, y))

# # Save results to a file
# np.save(filename, results)