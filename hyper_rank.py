import os
import string
from collections import OrderedDict
from time import sleep
import argparse

import networkx as nx
from nltk.stem import PorterStemmer
from tqdm.auto import tqdm
import numpy as np

from tree_rep import HyperbolicEmbedding
from utils import read_json, merge_dicts
from graph_construction import ConstructGraph
import stanza
import pickle

from evaluate import evaluate_model


class HyperRank:
    def __init__(self, parameter_file, data, use_stanza=False, use_gpu=False):
        if isinstance(parameter_file, str):
            self.params = read_json(parameter_file)
        else:
            self.params = parameter_file

        self.params["path"] = data

        self.path_to_test = os.path.join(self.params["path"], 'test.json')
        self.path_to_train = os.path.join(self.params["path"], 'train.json')

        self.params["gpu"] = use_gpu

        self.stopwords = []
        with open(r"data/stopwords_en_yake.txt", 'r', encoding="utf8") as File:
            for line in File.readlines():
                self.stopwords.append(str(line).strip())

        self.ps = PorterStemmer()

        self.stanza = use_stanza

        if self.stanza:
            self.nlp = self.__load_stanza()

    def __load_stanza(self):
        preprocessors = 'tokenize,mwt,pos,lemma,depparse'

        return stanza.Pipeline(lang='en', processors=preprocessors, use_gpu=self.params['gpu'])

    def texttofreq(self, wordstring):
        words = wordstring.lower().split()
        wordlist = [self.ps.stem(word) for word in words]
        wordfreq = []
        for w in wordlist:
            wordfreq.append(wordlist.count(w))
        return dict(zip(wordlist, wordfreq))

    def run(self, train_files=False):
        # container for keyphrases
        keyphrases = {}
        stemmed_keyphrases = {}

        # get class from module
        texts = read_json(self.path_to_test)

        if train_files:
            texts = merge_dicts(read_json(self.path_to_train), texts)

        if not self.stanza:
            if os.path.exists(os.path.join(self.params["path"], 'graph.pkl')):
                with open(os.path.join(self.params["path"], 'graph.pkl'), 'rb') as f:
                    pickle_file = pickle.load(f)
            else:
                self.stanza = True

        # loop through the documents
        for file in tqdm(texts):
            text = texts[file]

            if self.stanza:
                cg = ConstructGraph(self.nlp)
                graph = cg.construct_graph(text)
            else:
                graph = nx.DiGraph(pickle_file[file])

            distance_matrix = nx.to_numpy_matrix(graph)

            he = HyperbolicEmbedding(distance_matrix, graph)
            new_graph = he.get_graph()

            # sleep(2)

            degrees = new_graph.degree

            my_degree = []
            for degree in degrees:
                if degree[1] > 0:
                    my_degree.append(degree)

            index = sorted(my_degree, key=lambda x: x[1], reverse=True)
            candidate_key = []
            for i in index:
                if isinstance(i[0], str):
                    word = i[0].split('.', 1)[0]
                    if word in set(self.stopwords) or word in string.punctuation or len(word) < 4:
                        continue
                    else:
                        candidate_key.append(word)

            unique_key = OrderedDict()
            for key in candidate_key:
                if self.ps.stem(key).lower() not in unique_key:
                    unique_key[self.ps.stem(key).lower()] = key

            inunique = {}
            freqtext = self.texttofreq(text)
            for i in freqtext.keys():
                if i in (unique_key.keys()):
                    inunique[i] = freqtext[i]
            finalunique = {k: v for k, v in sorted(inunique.items(), key=lambda item: item[1], reverse=True)}

            stemmed_keyphrases[file] = [[x] for x in list(finalunique.keys())]

            # stemmed_keyphrases[file] = [[x] for x in list(unique_key.keys())]
            keyphrases[file] = [[x] for x in list(unique_key.values())]

        return keyphrases, stemmed_keyphrases


def main(config):
    contains_train = {'500N-KPCrowd': True,
                      'DUC-2001': False}

    dataset = config['data']
    data_dir = os.path.join(config['data_dir'], dataset)
    use_stanza = config['use_stanza']
    gpu = config['gpu']
    run_on_train = config['train'] and contains_train[dataset]

    model = HyperRank({}, data_dir, use_stanza, gpu)
    keyphrases, stemmed_keyphrases = model.run(run_on_train)

    if config['eval']:
        path_ake_datasets = config['ake_dataset_path']

        dataset_annotator = {'500N-KPCrowd': 'reader',
                             'DUC-2001': 'reader'}

        ground_truth = read_json(os.path.join(path_ake_datasets,
                                              f'datasets/{dataset}/references/test.{dataset_annotator[dataset]}.stem.json'))
        if config['eval_train'] and contains_train[dataset]:
            train_gt = read_json(os.path.join(path_ake_datasets,
                                              f'datasets/{dataset}/references/train.{dataset_annotator[dataset]}.stem.json'))
            ground_truth = merge_dicts(ground_truth, train_gt)

        top_ten, top_five = False, False

        if config['top'] == "both":
            top_five = True
            top_ten = True
        elif config['top'] == '5':
            top_five = True
        else:
            top_ten = True

        if top_five:
            evaluate_model(stemmed_keyphrases, ground_truth, 5)
        if top_ten:
            evaluate_model(stemmed_keyphrases, ground_truth, 10)


if __name__ == '__main__':
    top_n = ("5", "10", "both")
    dataset = ["500N-KPCrowd",
               "DUC-2001"]

    parser = argparse.ArgumentParser('hyper_rank.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, default="500N-KPCrowd", choices=dataset, help="Dataset Name")
    parser.add_argument("--data-dir", type=str, default="data/", help="data directory")
    parser.add_argument("--ake-dataset-path", type=str, default='/home/duskybomb/GitProjects/ake-datasets/',
                        help="path to ake dataset")
    parser.add_argument("--use-stanza", action='store_true', help="Rerun stanza on the dataset")
    parser.add_argument("--gpu", action="store_true", help="use gpu for stanza")
    parser.add_argument("--train", action="store_true", help="Run on the training set")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--eval-train", action="store_true", help="Evaluate on the training set")
    parser.add_argument("--top", type=str, choices=top_n, help="evaluate on top n keywords")
    args = parser.parse_args()

    main(args.__dict__)
