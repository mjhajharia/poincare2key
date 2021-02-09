import argparse
import os
import pickle
import string
from collections import OrderedDict
from glob import glob

import networkx as nx
import stanza
from nltk.stem import PorterStemmer
from tqdm.auto import tqdm

from evaluate import evaluate_model
from graph_construction import ConstructGraph
from tree_rep import HyperbolicEmbedding
from utils import read_json, merge_dicts


class HyperRank:
    def __init__(self, parameter_file, data, use_stanza=False, use_gpu=False):
        self.params = parameter_file

        self.params["path"] = data

        self.path = {}

        for file in self.params["files"]:
            self.path[file.split('.')[0]] = os.path.join(self.params["path"], file)

        self.params["gpu"] = use_gpu

        self.stopwords = []
        with open(r"../data/stopwords_en_yake.txt", 'r', encoding="utf8") as File:
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

    def run(self, files):
        # container for keyphrases
        keyphrases = {}
        stemmed_keyphrases = {}

        # get class from module
        texts = read_json(self.path['test'])

        for file in files[1:]:
            texts = merge_dicts(read_json(self.path[file.split('.')[0]]), texts)

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
            keyphrases[file] = [[x] for x in list(unique_key.values())]

        return keyphrases, stemmed_keyphrases


def main(config):
    dataset = config['data']
    data_dir = os.path.join(config['data_dir'], dataset)
    use_stanza = config['use_stanza']
    gpu = config['gpu']

    dataset_details = read_json(os.path.join(config['data_dir'], f'{dataset}.json'))

    if config['files'] == "all":
        files = dataset_details['files']
    else:
        files = ["test"]

    model = HyperRank(dataset_details, data_dir, use_stanza, gpu)
    keyphrases, stemmed_keyphrases = model.run(files)

    with open(os.path.join(data_dir, "keyphrases.pkl"), "wb") as f:
        pickle.dump(stemmed_keyphrases, f)

    if 'eval' in config:
        ground_truth = read_json(os.path.join(config['data_dir'], dataset_details['references']['test']))
        if config['eval'] == "all":
            for file in files[1:]:
                ground_truth = merge_dicts(ground_truth,
                                           read_json(
                                               os.path.join(config['data_dir'],
                                                            dataset_details['references'][file.split(".")[0]])))

        for top in config['top']:
            evaluate_model(stemmed_keyphrases, ground_truth, top)


if __name__ == '__main__':
    top_n = ("5", "10", "both")
    dataset = [x.split('.')[0] for x in glob('../data/*.json')]
    files = ["test", "all"]

    parser = argparse.ArgumentParser('hyper_rank.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",
                        type=str,
                        default="500N-KPCrowd",
                        choices=dataset,
                        help="Dataset Name")
    parser.add_argument("--data-dir",
                        type=str,
                        default="../data/",
                        help="data directory")
    parser.add_argument("--use-stanza",
                        action='store_true',
                        help="Rerun stanza on the dataset")
    parser.add_argument("--gpu",
                        action="store_true",
                        help="use gpu for stanza")
    parser.add_argument("--files",
                        type=str,
                        choices=files,
                        default='all',
                        help="Run on files")
    parser.add_argument("--eval",
                        type=str,
                        choices=files,
                        default='all',
                        help="Evaluate the model")
    parser.add_argument("--top",
                        type=int,
                        nargs='+',
                        help="evaluate on top n keywords")
    args = parser.parse_args()

    main(args.__dict__)
