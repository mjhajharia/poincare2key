import string
from time import sleep

import networkx as nx
import numpy as np
from nltk.stem import PorterStemmer
from tqdm.auto import trange

from Hyperbolic import HyperbolicEmbedding
from evaluate import Evaluate
from graph_construction import LoadData

stopwords = []
with open(r"stopwords_en_yake.txt", 'r', encoding="utf8") as File:
    for line in File.readlines():
        stopwords.append(str(line)[:-1])

ps = PorterStemmer()


def evaluate_kp(data_set):
    config = {
        'dataset': data_set,
        'data_dir': 'data',
    }

    marujo_data = LoadData(config)

    metrics = {'precision': [],
               'recall': [],
               'f1': []}

    for item in trange(len(marujo_data.data)):
        data = marujo_data.data[item]['text']
        graph = marujo_data.construct_graph(data)

        distance_matrix = nx.to_numpy_matrix(graph)

        he = HyperbolicEmbedding(distance_matrix, graph)
        new_graph = he.get_graph()

        sleep(1)

        index = sorted(new_graph.degree, key=lambda x: x[1], reverse=True)
        candidate_key = []
        for i in index:
            if isinstance(i[0], str):
                word = i[0].split('.', 1)[0]
                if word in set(stopwords) or word in string.punctuation or len(word) < 4:
                    continue
                else:
                    candidate_key.append(word)

        unique_key = []
        for key in candidate_key:
            if ps.stem(key).lower() not in unique_key:
                unique_key.append(ps.stem(key).lower())

        if data_set == "marujo":
            keywords = marujo_data.data[item]['keywords'].split('\n')
        elif data_set == "hulth":
            keywords = [x.strip().replace('\n\t', ' ').replace('\n', ' ').replace('\t', ' ') for x in
                        marujo_data.data[item]['keywords'].split(';')]
        elif data_set == "sem-eval":
            keywords = [x for x in marujo_data.data[item]['keywords'].split('\n') if len(x) > 1]
        else:
            raise NotImplementedError

        ev = Evaluate('data', data_set)
        precision, recall, f1 = ev.evaluate_from_keyword(unique_key[:5], keywords)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

    return np.asarray(metrics['precision']).mean(), np.asarray(metrics['recall']).mean(), np.asarray(
        metrics['f1']).mean()
