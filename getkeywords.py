import networkx as nx
import numpy as np
import scipy.spatial as spatial
from nltk.stem import PorterStemmer

class getkeywords:
        
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
    
    def evaluate(candidate_key, keywords):
        ps = PorterStemmer()
        
        proposed = []
        groundtruth = []
        
        for i in candidate_key:
            proposed.append(ps.stem(i))
        for i in keywords:
            groundtruth.append(ps.stem(i))
            
        proposed_set = set(proposed)
        true_set = set(groundtruth)
        true_positives = len(proposed_set.intersection(true_set))
        if len(proposed_set)==0:
            precision = 0
        else:
            precision = true_positives/float(len(proposed)) 
        if len(true_set)==0:
            recall = 0
        else:
            recall = true_positives/float(len(true_set))
        if precision + recall > 0:
            f1 = 2*precision*recall/float(precision + recall)
        else:
            f1 = 0
        return (precision, recall, f1)
        
        