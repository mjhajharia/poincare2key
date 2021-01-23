import networkx as nx
import numpy as np
from scipy import spatial


class EdgeWeights:
    def __init__(self, graph, word_vector_map):
        self.graph = graph
        self.word_vector_map = word_vector_map
        self.num_nodes = len(self.graph.nodes())

    def calculate_edge_weights(self, vector_one_id, vector_two_id):

        return spatial.distance.cosine(
            self.word_vector_map[vector_one_id][0],
            self.word_vector_map[vector_two_id][0]
        )

    def insert_edge_weights(self):
        edgelist = list(self.graph.edges())
        for i in edgelist:   
            self.graph[i[0]][i[1]]['weight'] = self.calculate_edge_weights(i[0],i[1])

    def generate_distance(self):
        self.insert_edge_weights()
        dist_mat = np.zeros((self.num_nodes,self.num_nodes))
        listnodes = list(self.graph.nodes())

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if listnodes[j] in self.graph[listnodes[i]]:
                    dist_mat[i][j]= self.graph[listnodes[i]][listnodes[j]]['weight']
            
        return dist_mat
