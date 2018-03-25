import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from voronoi import create_grid_and_edges
import numpy.linalg as LA
from queue import PriorityQueue


def find_start_goal(nxg, start, goal):
    # TODO: find start and goal on skeleton
    # Some useful functions might be:
        # np.nonzero()
        # np.transpose()
        # np.linalg.norm()
        # np.argmin()
    skelCells = np.array(nxg.nodes())
    startMinDist = np.linalg.norm(np.array(start) - np.array(skelCells), axis=1).argmin()
    nearStart = skelCells[startMinDist]
    goalMinDist = np.linalg.norm(np.array(goal) - np.array(skelCells), axis=1).argmin()
    nearGoal = skelCells[goalMinDist]
    return list(nearStart), list(nearGoal)


def closest_point(graph, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    closest_point = None
    dist = 100000
    for p in graph.nodes:
        d = LA.norm(np.array(p) - np.array(current_point))
        if d < dist:
            closest_point = p
            dist = d
    return closest_point


def heuristic(n1, n2):
    # TODO: define a heuristic
    return LA.norm(np.array(n1) - np.array(n2))


def get_next_nodes(g, current_node):
    nextNodes = []
    for neighbor in g.neighbors(current_node):
        edata = g.get_edge_data(current_node, neighbor)
        nextNodes.append({'node':neighbor,
                          'cost':edata['weight']})
    return nextNodes


###### THIS IS YOUR OLD GRID-BASED A* IMPLEMENTATION #######
###### With a few minor modifications it can work with graphs! ####
# TODO: modify A* to work with a graph
def a_star(graph, heuristic, start, goal):
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = tuple(item[1])

        if current_node == goal:
            print('Found a path.')
            found = True
            break

        else:
            for nextNode in get_next_nodes(graph, current_node):
                # get the tuple representation
                cost = nextNode['cost']
                new_cost = current_cost + cost + heuristic(nextNode['node'], goal)

                if nextNode['node'] not in visited:
                    visited.add(nextNode['node'])
                    queue.put((new_cost, nextNode['node']))

                    branch[nextNode['node']] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost


def create_weighted_graph(edges):
    G = nx.Graph()
    for p1, p2 in edges:
        dist = LA.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)
    return G


if __name__ == "__main__":
    # This is the same obstacle data from the previous lesson.
    filename = 'data/colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    start_ne = (25, 100)
    goal_ne = (750., 370.)

    # Static drone altitude (metres)
    drone_altitude = 5
    safety_distance = 3

    # This is now the routine using Voronoi
    grid, edges = create_grid_and_edges(data, drone_altitude, safety_distance)
    print(len(edges))

    # equivalent to
    # plt.imshow(np.flip(grid, 0))
    plt.imshow(grid, origin='lower', cmap='Greys')

    for e in edges:
        p1 = e[0]
        p2 = e[1]
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

    plt.plot(start_ne[1], start_ne[0], 'rx')
    plt.plot(goal_ne[1], goal_ne[0], 'rx')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()

    weightedGraph = create_weighted_graph(edges)
    start1, goal1 = find_start_goal(weightedGraph, start_ne, goal_ne)
    print(start1, goal1)
    # start = closest_point(weightedGraph, start_ne)
    # goal = closest_point(weightedGraph, goal_ne)
    # print(start, goal)
    path, cost = a_star(weightedGraph, heuristic, tuple(start1), tuple(goal1))
    print(len(path), cost)

    # equivalent to
    # plt.imshow(np.flip(grid, 0))
    plt.imshow(grid, origin='lower', cmap='Greys')

    for e in edges:
        p1 = e[0]
        p2 = e[1]
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

    plt.plot([start_ne[1], start1[1]], [start_ne[0], start1[0]], 'r-')
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')
    plt.plot([goal_ne[1], goal1[1]], [goal_ne[0], goal1[0]], 'r-')

    plt.plot(start_ne[1], start_ne[0], 'gx')
    plt.plot(goal_ne[1], goal_ne[0], 'gx')

    plt.xlabel('EAST', fontsize=20)
    plt.ylabel('NORTH', fontsize=20)
    plt.show()


