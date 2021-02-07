import julia

julia.install()
from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

Main.eval('include("TreeRep/TreeRep.jl")')
Main.eval('using LightGraphs')


class HyperbolicEmbedding:
    def __init__(self, distance_matrix, graph):
        self.distance_matrix = distance_matrix
        self.graph = graph

    def get_tree_rep(self):
        Main.d = self.distance_matrix

        Main.G, Main.dist = Main.eval("TreeRep.metric_to_structure(d)")
        edges = Main.eval('collect(edges(G))')
        n = Main.eval('nv(G)')
        W = np.zeros((n, n))  # Initializing the adjacency matrix
        for edge in edges:
            src = edge.src - 1
            dst = edge.dst - 1
            W[src, dst] = Main.dist[src, dst]
            W[dst, src] = Main.dist[dst, src]

        return W

    def get_graph(self, relabel=True):
        W = self.get_tree_rep()

        newW = W[0:W.shape[0], 0:W.shape[0]]
        G = nx.from_numpy_matrix(np.array(newW), create_using=nx.Graph)

        # Relabel Graph
        if relabel:
            res = {}
            test_keys = list(G.nodes())
            test_values = list(self.graph.nodes())
            for key in test_keys:
                for value in test_values:
                    res[key] = value
                    test_values.remove(value)
                    break

            return nx.relabel_nodes(G, res)
        return G

    def visualize(self, output_file='graph.svg'):
        fig = plt.figure(figsize=(120, 120))
        nx.draw(self.get_graph(), node_size=30, with_labels=True)
        plt.axis('equal')
        plt.show()
        fig.savefig('graph.svg')
