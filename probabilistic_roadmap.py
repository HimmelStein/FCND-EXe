import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
from configuration_space import create_grid
from random_sampling import extract_polygons
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue


def heuristic(n1, n2):
    # TODO: complete
    return 0


def a_star(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""

    # TODO: complete
    return []


def can_connect(pt1, pt2, polygons):
    ln = LineString([pt1[:2], pt2[:2]])
    for (polygon, h) in polygons:
        if polygon.crosses(ln) and h <= min(pt1[2], pt2[2]):
            return False
    return True


def create_graph(nodes, k, polygons):
    # for each node connect try to connect to k nearest nodes
    g = nx.Graph()
    tree = KDTree(nodes)
    for nd in nodes:
        idxs = tree.query([nd], k, return_distance=False)[0]
        for idx in idxs:
            n2 = nodes[idx]
            if n2 == nd:
                continue
            if can_connect(nd, n2, polygons):
                g.add_edge(nd, n2, weight=1)
    return g


if __name__ == "__main__":
    # This is the same obstacle data from the previous lesson.
    filename = 'data/colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    print(data)

    polygons = extract_polygons(data)

    # sample points
    # TODO: sample points randomly
    # then use KDTree to find nearest neighbor polygon
    # and test for collision

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
    nodes = samples[:10]

    # TODO: connect nodes
    # Suggested method
    # 1) cast nodes into a graph called "g" using networkx

    # 2) write a method "can_connect()" that:
    #       casts two points as a shapely LineString() object
    #       tests for collision with a shapely Polygon() object
    #       returns True if connection is possible, False otherwise

    # 3) write a method "create_graph()" that:
    #       defines a networkx graph as g = Graph()
    #       defines a tree = KDTree(nodes)
    #       test for connectivity between each node and
    #           k of it's nearest neighbors
    #       if nodes are connectable, add an edge to graph

    # Iterate through all candidate nodes!

    g = create_graph(nodes, 10, polygons)

    # Create a grid map of the world


    # This will create a grid map at 1 m above ground level
    grid = create_grid(data, 1, 1)

    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])

    # If you have a graph called "g" these plots should work
    # Draw edges
    for (n1, n2) in g.edges:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black' , alpha=0.5)

    # Draw all nodes connected or not in blue
    for n1 in nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')

    # Draw connected nodes in red
    for n1 in g.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')

    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()

    ## Step 7 - Visualize Path

    fig = plt.figure()

    plt.imshow(grid, cmap='Greys', origin='lower')

    # Add code to visualize path here

    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()