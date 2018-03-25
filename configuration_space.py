import numpy as np
import matplotlib.pyplot as plt


"""
confguration Space

In this notebook you'll create a configuration space given a map of the world and setting a particular altitude for 
your drone. You'll read in a .csv file containing obstacle data which consists of six columns  xx ,  yy ,  zz  
and  δxδx ,  δyδy ,  δzδz .

You can look at the .csv file here. The first line gives the map center coordinates and the file is arranged such that:

xx  -> NORTH
yy  -> EAST
zz  -> ALTITUDE (positive up, note the difference with NED coords)
Each  (x,y,z)(x,y,z)  coordinate is the center of an obstacle.  δxδx ,  δyδy ,  δzδz  are the half widths of the 
obstacles, meaning for example that an obstacle with  (x=37,y=12,z=8)(x=37,y=12,z=8)  and  
(δx=5,δy=5,δz=8)(δx=5,δy=5,δz=8) is a 10 x 10 m obstacle that is 16 m high and is centered at the point  
(x,y)=(37,12)(x,y)=(37,12)  at a height of 8 m.

Given a map like this, the free space in the  (x,y)(x,y)  plane is a function of altitude, and you can plan a path 
around an obstacle, or simply fly over it! You'll extend each obstacle by a safety margin to create the equivalent 
of a 3 dimensional configuration space.

Your task is to extract a 2D grid map at 1 metre resolution of your configuration space for a particular altitude, 
where each value is assigned either a 0 or 1 representing feasible or infeasible (obstacle) spaces respectively.

The end result should look something like this ... (colours aren't important)
"""


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = int(np.floor(np.amin(data[:, 0] - data[:, 3])))
    north_max = int(np.ceil(np.amax(data[:, 0] + data[:, 3])))

    # minimum and maximum east coordinates
    east_min = int(np.floor(np.amin(data[:, 1] - data[:, 4])))
    east_max = int(np.ceil(np.amax(data[:, 1] + data[:, 4])))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        # TODO: Determine which cells contain obstacles
        # and set them to 1.
        #
        # Example:
        #
        #    grid[north_coordinate, east_coordinate] = 1
        if drone_altitude - alt - d_alt - safety_distance < 0:
            north_0 = np.clip(int(north - d_north - safety_distance - north_min), 0, north_size)
            north_1 = np.clip(int(north + d_north + safety_distance - north_min), 0, north_size)
            east_0 = np.clip(int(east - d_east - safety_distance - east_min), 0, east_size)
            east_1 = np.clip(int(east + d_east + safety_distance - east_min), 0, east_size)
            grid[north_0:north_1, east_0:east_1] = 1

    return grid


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = [12, 12]
    filename = 'data/colliders.csv'
    # Read in the data skipping the first two lines.
    # Note: the first line contains the latitude and longitude of map center
    # Where is this??

    data = np.loadtxt(filename,delimiter=',',dtype='Float64',skiprows=2)
    print(data)

    # Static drone altitude (metres)
    drone_altitude = 5

    # Minimum distance required to stay away from an obstacle (metres)
    # Think of this as padding around the obstacles.
    safe_distance = 3

    grid = create_grid(data, drone_altitude, safe_distance)

    # equivalent to
    # plt.imshow(np.flip(grid, 0))
    # NOTE: we're placing the origin in the lower lefthand corner here
    # so that north is up, if you didn't do this north would be positive down
    plt.imshow(grid, origin='lower')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()