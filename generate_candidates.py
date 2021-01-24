import networkx as nx
import numpy as np
from scipy import spatial


def candidates(graph):
    pos = nx.spring_layout(graph)
    val = list(pos.values())
    x = []
    y = []
    for i in val:
        x.append(i[0])
        y.append(i[1])
    coordinate = np.column_stack((x, y))
    point_tree = spatial.cKDTree(coordinate)
    r = 0.5
    index = point_tree.query_ball_point([0.0, 0.0], r)
    wordlist= list(pos.keys())
    candidate_key = []
    for i in index:
        candidate_key.append(wordlist[i].split('.',1)[0])  
    stopwords = []
    with open(r"stopwords.txt",'r', encoding="utf8") as File:
        for line in File.readlines():
            stopwords.append(str(line)[:-1])
        for i in candidate_key:
            if i in set(stopwords):
                candidate_key.remove(i)
    return candidate_key