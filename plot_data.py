import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Get list of all .npy files in the data directory
data_files = glob.glob('data/*.npy')

# Find the latest file based on modification time
latest_file = max(data_files, key=os.path.getmtime)

# Load the latest data file
data = np.load(latest_file)
x = data[:, 0]
y = data[:, 1]

# Plot the data
plt.plot(x, y)
plt.xlabel('Fermi Radius N')
plt.ylabel('S(k, N)')
plt.show()