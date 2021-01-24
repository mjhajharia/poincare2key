import os

import networkx as nx
import numpy as np
import scipy.spatial as spatial
from nltk.stem import PorterStemmer
from tqdm import tqdm

from graph_construction import data_json_path
from utils import read_json


class Evaluate:
    def __init__(self, data_dir="", data_set=None):
        if data_set is not None:
            self.data_path = data_json_path[data_set]
            self.data_dir = data_dir

            self.data = read_json(os.path.join(self.data_dir, self.data_path))

    def evaluate(self, method, top=5, item=None):
        assert method in ('topicrank', 'textrank',
                          'positionrank', 'multipartiterank')

        assert top in (5, 10)

        if top == 5:
            top = 0
        else:
            top = 1

        if item is None:
            metrics = {'precision': [],
                       'recall': [],
                       'f1': []}
            for data in tqdm(self.data, total=len(self.data)):
                keywords = data['keywords'].split('\n')
                candidate = data[method][top]
                precision, recall, f1 = self.evaluate_from_keyword(candidate, keywords)

                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)

            return np.asarray(metrics['precision']).mean(), np.asarray(metrics['recall']).mean(), np.asarray(metrics['f1']).mean()
        else:
            assert item < len(self.data) 

            keywords = self.data[item]['keywords'].split('\n')
            candidate = self.data[item][method][top]
            # print(keywords)
            # print(candidate)
            return self.evaluate_from_keyword(candidate, keywords)


    @staticmethod
    def evaluate_from_keyword(candidate_key, keywords):
        ps = PorterStemmer()

        proposed = []
        groundtruth = []

        for i in candidate_key:
            proposed.append(ps.stem(i).lower())
        for i in keywords:
            groundtruth.append(ps.stem(i).lower())

        proposed_set = set(proposed)
        true_set = set(groundtruth)
        true_positives = len(proposed_set.intersection(true_set))

        if len(proposed_set) == 0:
            precision = 0
        else:
            precision = true_positives/float(len(proposed))

        if len(true_set) == 0:
            recall = 0
        else:
            recall = true_positives/float(len(true_set))

        if precision + recall > 0:
            f1 = 2*precision*recall/float(precision + recall)
        else:
            f1 = 0

        return (precision, recall, f1)
