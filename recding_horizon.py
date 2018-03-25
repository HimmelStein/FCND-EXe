import numpy as np
import matplotlib.pyplot as plt

# Grid creation routine
from configuration_space import create_grid
# Voxel map creation routine
from threeD_map import create_voxmap
# 2D A* planning routine (can you convert to 3D??)
from planning import a_star
# Random sampling routine


# This notebook is your playground to pull together techniques from the previous lessons!
# A solution here can be built from previous solutions (more or less) so we will offer no
# solution notebook this time.

# Here's a suggested approach:

# Load the colliders data
# Discretize your search space into a grid or graph
# Define a start and goal location
# Find a coarse 2D plan from start to goal
# Choose a location along that plan and discretize a local volume around that location (for example, you might try
# a 40x40 m area that is 10 m high discretized into 1m^3 voxels)
# Define your goal in the local volume to a a node or voxel at the edge of the volume in the direction of the next
# waypoint in your coarse global plan.
# Plan a path through your 3D grid or graph to that node or voxel at the edge of the local volume.
# We'll import some of the routines from previous exercises that you might find useful here.


if __name__ == "__main__":
    # This is the same obstacle data from the previous lesson.
    filename = 'data/colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    flight_altitude = 3
    safety_distance = 3
    grid = create_grid(data, flight_altitude, safety_distance)

    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()