import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Number of files to plot
total_files = 5

# Create directory if not exists
os.makedirs('data', exist_ok=True)

# Get list of all .npy files in the data directory
data_files = glob.glob('data/*.npy')

# Load last series of files
file_series = data_files[-total_files-1:]
file_series.sort(key=os.path.getmtime, reverse=True)

# Initialize the plot
plt.figure()

for file in file_series:
    data = np.load(file)
    x = data[:, 0]
    y = data[:, 1]

    # Plot the data
    plt.plot(x, y, label=os.path.basename(file))

# Add labels and legend
plt.xlabel('Fermi Radius N')
plt.ylabel('S(k, N)')
plt.legend()
plt.savefig('combined_plot.png')
plt.show()