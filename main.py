# Run the simulation and plot the results.
from BaseLune import Resolvent, prime_coprime_pair, ResolventGraph
import numpy as np
import time
import matplotlib.pyplot as plt

#####################
# Simulation Params #
#####################
N = 110 # Fermi Radius ( \geq 100 )
L = 6   # x lattice cutoff
#####################

# Create unique filename based on k, N, and current time
current_time = time.strftime("%Y-%m-%d-%H-%M-%S")

# Initialize the plot
plt.figure()


for i in range(L):
    for j in range(i+1):
        k = np.array([j,i])
        filename = f'data/k=({i,j})_N={N}_resolvent_{time.strftime("%Y-%m-%d-%H-%M-%S")}.npy'
        print(f"k = {k}, N = {N}")
        x,y = ResolventGraph(k, N)
        data = np.column_stack((x, y))
        plt.plot(x, y, label=f'k = {k}')
        np.save(filename, data)

# Adjust the legend
plt.legend(bbox_to_anchor=(0, 0), loc='lower right', borderaxespad=0.)
plt.tight_layout()  # Adjust the layout to prevent overlapping
plt.savefig('combined_plot.png', bbox_inches='tight')  # Save with tight bounding box

# Add labels and legend
plt.xlabel(f'Fermi Radius k_F')
plt.ylabel('S(k)')
plt.legend()
plt.savefig('combined_plot.png')
plt.show()


# # Save the function to a file
# k = np.array([5,7])
# x = np.arange(100, 200, 1)
# x,y = ResolventGraph(x, k, N=200)
# data = np.column_stack((x, y))
# np.save(filename, data)