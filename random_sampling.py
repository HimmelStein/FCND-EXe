import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, Point
from configuration_space import create_grid


def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        # TODO: Extract the 4 corners of the obstacle
        #
        # NOTE: The order of the points matters since
        # `shapely` draws the sequentially from point to point.
        #
        # If the area of the polygon is 0 you've likely got a weird
        # order.
        corners = [(north - d_north, east - d_east),
                   (north - d_north, east + d_east),
                   (north + d_north, east + d_east),
                   (north + d_north, east - d_east)]

        # TODO: Compute the height of the polygon
        height = d_alt + alt

        # TODO: Once you've defined corners, define polygons
        p = Polygon(corners)
        polygons.append((p, height))

    return polygons


# Removing Points Colliding With Obstacles
# Prior to remove a point we must determine whether it collides with any obstacle.
# Complete the collides function below.
# It should return True if the point collides with any obstacle and False if no collision is detected.
def collides(polygons, point):
    # TODO: Determine whether the point collides
    # with any obstacles.
    for polygon, h in polygons:
        point0 = Point(tuple(point[:2]))
        if polygon.contains(point0) and h < point[2]:
            return True
    return False


if __name__ == "__main__":
    # This is the same obstacle data from the previous lesson.
    filename = 'data/colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    polygons = extract_polygons(data)

    # Now that we have the extracted the polygons, we need to sample random 3D points. Currently we don't know suitable
    # ranges for x, y, and z. Let's figure out the max and min values for each dimension.
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    zmin = 0
    # Limit the z axis for the visualization
    zmax = 10

    print("X")
    print("min = {0}, max = {1}\n".format(xmin, xmax))

    print("Y")
    print("min = {0}, max = {1}\n".format(ymin, ymax))

    print("Z")
    print("min = {0}, max = {1}".format(zmin, zmax))

    # Next, it's time to sample points. All that's left is picking the distribution and number of samples.
    # The uniform distribution makes sense in this situation since we we'd like to encourage searching the whole space.
    num_samples = 100

    xvals = np.random.uniform(xmin, xmax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(zmin, zmax, num_samples)

    samples = list(zip(xvals, yvals, zvals))
    samples[:10]

    t0 = time.time()
    to_keep = []
    for point in samples:
        if not collides(polygons, point):
            to_keep.append(point)
    time_taken = time.time() - t0
    print("Time taken {0} seconds ...", time_taken)
    print(len(to_keep))

    grid = create_grid(data, zmax, 1)
    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    # draw points
    all_pts = np.array(to_keep)
    north_vals = all_pts[:, 0]
    east_vals = all_pts[:, 1]
    plt.scatter(east_vals - emin, north_vals - nmin, c='red')

    plt.ylabel('NORTH')
    plt.xlabel('EAST')

    plt.show()