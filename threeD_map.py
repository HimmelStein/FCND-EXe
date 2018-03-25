import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_voxmap(data, voxel_size=5):
    """
    Returns a grid representation of a 3D configuration space
    based on given obstacle data.

    The `voxel_size` argument sets the resolution of the voxel map.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))

    alt_max = np.ceil(np.amax(data[:, 2] + data[:, 5]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min))) // voxel_size
    east_size = int(np.ceil((east_max - east_min))) // voxel_size
    alt_size = int(alt_max) // voxel_size

    voxmap = np.zeros((north_size, east_size, alt_size), dtype=np.bool)

    for i in range(data.shape[0]):
        # TODO: fill in the voxels that are part of an obstacle with `True`
        #
        # i.e. grid[0:5, 20:26, 2:7] = True
        north0 = int((data[i][0] - data[i][3] - north_min) / voxel_size)
        north1 = int(np.ceil((data[i][0] + data[i][3] - north_min) / voxel_size))
        east0 = int((data[i][1] - data[i][4] - east_min) / voxel_size)
        east1 = int(np.ceil((data[i][1] + data[i][4] - east_min) / voxel_size))
        alt1 = int(np.ceil((data[i][2] + data[i][5]) / voxel_size))
        voxmap[north0:north1, east0:east1, 0:alt1] = True

    return voxmap


def plot_3d(voxmap):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlim(voxmap.shape[0], 0)
    ax.set_ylim(0, voxmap.shape[1])
    # add 100 to the height so the buildings aren't so tall
    ax.set_zlim(0, voxmap.shape[2] + 70)

    ax.voxels(voxmap, edgecolor='k')

    plt.xlabel('North')
    plt.ylabel('East')
    plt.show()


if __name__ == "__main__":
    # This is the same obstacle data from the previous lesson.
    filename = 'data/colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    voxmap = create_voxmap(data, voxel_size=10)
    print(voxmap.shape)
    plot_3d(voxmap)