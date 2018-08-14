from queue import Queue
from heaps import PriorityQueue
import sys


class Vertex(object):

    def __init__(self, key, color='white'):
        self.id = key
        self.connections = {}
        self._color = color

        self._distance = 0
        self._predecessor = None
        self._discovery = None
        self._finish = None

    def __str__(self):
        return str(self.id) + ' connected to: ' + str([x.id for x in self.connections])

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, v):
        if v in ['white', 'gray', 'black']:
            self._color = v
        else:
            raise ValueError('Invalid color.')

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, d):
        self._distance = d

    @property
    def discovery(self):
        return self._discovery

    @discovery.setter
    def discovery(self, time):
        self._discovery = time

    @property
    def finish(self):
        return self._finish

    @finish.setter
    def finish(self, time):
        self._finish = time

    def add_neighbor(self, nbr, weight=0):
        self.connections[nbr] = weight

    def get_connections(self):
        return self.connections.keys()

    def get_weight(self, nbr):
        return self.connections[nbr]


class Graph(object):
    """Adjacency list implementation of a graph."""

    def __init__(self):
        self.vertices = {}

    @property
    def num_vertices(self):
        return len(self.vertices)

    def __contains__(self, k):
        return k in self.vertices

    def __iter__(self):
        return iter(self.vertices.values())

    def add_vertex(self, key):
        self.vertices[key] = Vertex(key)
        return self.vertices[key]

    def get_vertex(self, key):
        return self.vertices.get(key)

    def add_edge(self, i, j, weight):
        if i not in self.vertices:
            self.add_vertex(i)
        if j not in self.vertices:
            self.add_vertex(j)

        self.vertices[i].add_neighbor(j, weight)

    def get_vertices(self):
        return self.vertices.keys()


class BFSGraph(Graph):
    def __init__(self):
        super(BFSGraph, self).__init__()

    def bfs(self, start_vertex):
        start_vertex.distance = 0
        start_vertex.predecessor = None

        vertex_queue = Queue()
        vertex_queue.enqueue(start_vertex)

        while vertex_queue.size() > 0:
            current_vertex = vertex_queue.dequeue()
            for neighbor in current_vertex.get_connections():
                if neighbor.color == 'white':
                    neighbor.distance = current_vertex.distance + 1
                    neighbor.color = 'gray'
                    neighbor.predecessor = current_vertex
                    vertex_queue.enqueue(neighbor)

                current_vertex.color = 'black'

    def traverse(self, vertex):
        v = vertex

        while v.predecessor is not None:
            print(v.id)
            v = v.predecessor

        print(v.id)


class DFSGraph(Graph):
    def __init__(self):
        super(DFSGraph, self).__init__()
        self.time = 0

    def dfs(self):
        for vertex in self:
            vertex.color = 'white'
            vertex.predecessor = -1

        for vertex in self:
            if vertex.color == 'white':
                self.traverse(vertex)

    def traverse(self, vertex):
        vertex.color = 'gray'
        self.time += 1
        vertex.discovery = self.time

        for nbr in vertex.get_connections():
            if nbr.color == 'white':
                nbr.predecessor = vertex
                self.traverse(nbr)

        vertex.color = 'black'
        self.time += 1
        vertex.finish = self.time


def shortest_path(graph, source_node):
    """Dijkstra's Algorithm"""

    for node in graph:
        node.distance = float("inf")
        node.predecessor = None

    pq = PriorityQueue()
    source_node.distance = 0
    pq.build_heap([(v.distance, v) for v in graph])

    while not pq.is_empty():
        current_node = pq.del_min()
        for next_node in current_node.get_connections():
            new_dist = current_node.distance + \
                current_node.get_weight(next_node)

            if new_dist < next_node.distance:
                next_node.distance = new_dist
                next_node.predecessor = current_node
                pq.decrease_key(next_node, new_dist)

    return graph


def min_spanning_tree(graph, source_node):
    """Prim's minimum spanning tree algorithm"""
    for node in graph:
        node.distance = float("inf")
        node.predecessor = None

    pq = PriorityQueue()
    source_node.distance = 0
    pq.build_heap([(v.distance, v) for v in graph])

    while not pq.is_empty():
        current_node = pq.del_min()
        for next_node in current_node.get_connections():
            new_cost = current_node.get_weight(next_node)

            if next_node in pq and new_cost < next_node.distance:
                next_node.distance = new_cost
                next_node.predecessor = current_node
                pq.decrease_key(next_node, new_cost)

    return graph
