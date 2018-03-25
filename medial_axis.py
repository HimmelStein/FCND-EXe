import numpy as np
import matplotlib.pyplot as plt
from configuration_space import create_grid
from skimage.morphology import medial_axis
from skimage.util import invert
from planning import a_star

# TODO: Your start and goal location defined above
# will not necessarily be on the skeleton so you
# must first identify the nearest cell on the
# skeleton to start and goal


def find_start_goal(skel, start, goal):
    # TODO: find start and goal on skeleton
    # Some useful functions might be:
        # np.nonzero()
        # np.transpose()
        # np.linalg.norm()
        # np.argmin()
    skelCells = np.transpose(skel.nonzero())
    startMinDist = np.linalg.norm(np.array(start) - np.array(skelCells), axis=1).argmin()
    nearStart = skelCells[startMinDist]
    goalMinDist = np.linalg.norm(np.array(goal) - np.array(skelCells), axis=1).argmin()
    nearGoal = skelCells[goalMinDist]
    return nearStart, nearGoal


def heuristic_func(position, goal_position):
    # TODO: define a heuristic
    h = np.sqrt((position[0] - goal_position[0]) ** 2 + (position[1] - goal_position[1]) ** 2)
    return h


if __name__ == "__main__":
    filename = 'data/colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)
    start_ne = (25, 100)
    goal_ne = (650, 500)

    # Static drone altitude (meters)
    drone_altitude = 5
    safety_distance = 2

    grid = create_grid(data, drone_altitude, safety_distance)
    skeleton = medial_axis(invert(grid))

    # equivalent to
    # plt.imshow(np.flip(grid, 0))

    plt.imshow(grid, cmap='Greys', origin='lower')
    plt.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)

    plt.plot(start_ne[1], start_ne[0], 'rx')
    plt.plot(goal_ne[1], goal_ne[0], 'rx')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()

    skel_start, skel_goal = find_start_goal(skeleton, start_ne, goal_ne)

    print(start_ne, goal_ne)
    print(skel_start, skel_goal)

    # Run A* on the skeleton
    path, cost = a_star(invert(skeleton).astype(np.int), heuristic_func, tuple(skel_start), tuple(skel_goal))
    print("Path length = {0}, path cost = {1}".format(len(path), cost))

    # Compare to regular A* on the grid
    path2, cost2 = a_star(grid, heuristic_func, start_ne, goal_ne)
    print("Path length = {0}, path cost = {1}".format(len(path2), cost2))

    plt.imshow(grid, cmap='Greys', origin='lower')
    plt.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
    # For the purposes of the visual the east coordinate lay along
    # the x-axis and the north coordinates long the y-axis.
    plt.plot(start_ne[1], start_ne[0], 'x')
    # Uncomment the following as needed
    plt.plot(goal_ne[1], goal_ne[0], 'x')

    # pp = np.array(path)
    # plt.plot(pp[:, 1], pp[:, 0], 'g')

    pp2 = [skel_start]
    for pos in path2:
        curPos = pp2[-1]
        da = pos.value
        pp2= np.vstack([pp2, [curPos[0]+da[0], curPos[1]+da[1]]])
    plt.plot(pp2[:, 1], pp2[:, 0], 'r')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()
