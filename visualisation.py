import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

# Read data from csv file
data = []
path = "visualisation_data.csv"
with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        data.append(row)

# Get arrays dimension and separate data
dimension, data = int(data[0][0]), data[1]
data = data[:-1]
separated_data = [np.array(data[a * int(len(data) / 4):(a + 1) * int(len(data) / 4)],
                           dtype=np.float).reshape((dimension, dimension)) for a in range(4)]

# Plot 2D visualisation
fig, axs = plt.subplots(2, 2)
data = np.vstack(separated_data)
norm = Normalize(vmin=np.min(data), vmax=np.max(data))

# Plot 4 states in 2D
for i, idx in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    pl = axs[idx].imshow(separated_data[i][:-1, :], norm=norm, cmap=plt.get_cmap('hot'), interpolation="bicubic")

# Add title
fig.suptitle('2D Visualisation', fontsize=16)

# Disable axises
for ax in axs.flat:
    ax.set_axis_off()

# Add color bar
color_bar = fig.add_axes([0.9, 0.15, 0.03, 0.7])
color_bar.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(pl, cax=color_bar)

# Plot 3D visualisation of final state
fig2 = plt.figure(2)
ax = fig2.gca(projection='3d')
x, y = np.mgrid[0:dimension, 0:dimension]
ax.plot_surface(x, y, separated_data[3][x, y], cmap="hot")

# Add title and labels
fig2.suptitle('3D Visualisation', fontsize=16)
ax.set_xlabel("Y Axis")
ax.set_ylabel("X Axis")
ax.set_zlabel("Temperature")

plt.show()
