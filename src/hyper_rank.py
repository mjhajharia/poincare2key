import argparse
import os
import pickle
import string
from collections import OrderedDict
from glob import glob
import warnings
from datetime import datetime
import json

import networkx as nx
import stanza
from nltk.stem import PorterStemmer
from tqdm.auto import tqdm
import spacy
from spacy.tokenizer import Tokenizer
import textacy

from evaluate import evaluate_model
import graph_construction_chunks
import graph_construction
from tree_rep import HyperbolicEmbedding
from utils import read_json, merge_dicts, texttofreq
from rank_candidates import rank

warnings.filterwarnings("ignore", category=DeprecationWarning)


class HyperRank:
    def __init__(self, parameter_file, data, n_grams, use_stanza=False, use_spacy=False, use_gpu=False):
        # set parameters from files inside each dataset folder (dataset_detial)
        self.params = parameter_file

        # path to the dataset
        self.params["path"] = data

        # get path for all the files eg: test, train, val
        self.path = {}

        for file in self.params["files"]:
            self.path[file.split('.')[0]] = os.path.join(self.params["path"], file)

        self.params["gpu"] = use_gpu

        # load stopwords
        self.stopwords = []
        with open(r"../data/stopwords_en_yake.txt", 'r', encoding="utf8") as File:
            for line in File.readlines():
                self.stopwords.append(str(line).strip())

        self.ps = PorterStemmer()

        self.en = textacy.load_spacy_lang("en_core_web_lg", disable=("parser",))
        infixes = tuple([r"'s\b", r"(?<!\d)\.(?!\d)"]) + self.en.Defaults.prefixes
        infix_re = spacy.util.compile_infix_regex(infixes)

        self.en.tokenizer = self.__custom_tokenizer(infix_re)

        self.stanza = use_stanza
        self.spacy = use_spacy

        if self.stanza:
            self.nlp = self.__load_stanza()
        if self.spacy:
            self.nlp = spacy.load("en_core_web_lg")

        self.n_grams = n_grams

    def __custom_tokenizer(self, infix_re):
        return Tokenizer(self.en.vocab, infix_finditer=infix_re.finditer)

    def __load_stanza(self):
        preprocessors = 'tokenize,mwt,pos,lemma,depparse'

        return stanza.Pipeline(lang='en', processors=preprocessors, use_gpu=self.params['gpu'])

    def generate_candidates(self, text, keyphrases):
        doc = textacy.make_spacy_doc(text, lang=self.en)

        candidate_type_1 = textacy.extract.matches(doc,
                                                   [{"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}},
                                                    {"OP": "?", "POS": {"IN": ["NOUN", "PROPN", "ADJ"]}},
                                                    {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}, "OP": "+"}])

        candidate_type_1 = [x.text for x in candidate_type_1]

        chunks = list(set(candidate_type_1))

        candidates = []

        for key in keyphrases:
            for chunk in chunks:
                if key.lower() in [x.lower() for x in " ".join(chunk.split("-")).split()]:
                    candidates.append(chunk.lower())

        # candidates = list(set(candidates + [x.lower() for x in keyphrases]))
        candidates = list(set(candidates))

        final_candidates = []

        for candidate in candidates:
            change = False

            if "(" in candidate:
                new = candidate.split("(")
                new = [x.replace(")", "") for x in new]
                change = True
            if '"' in candidate:
                new = candidate.split('"')
                change = True
            if '[' in candidate:
                new = candidate.split('[')
                new = [x.replace("]", "") for x in new]
                change = True

            if not change:
                new = [candidate]

            final_candidates.extend(new)

        return final_candidates

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

        all_uniques = {}
        # loop through the documents
        for file in tqdm(texts):
            text = texts[file]

            if self.stanza:
                cg = graph_construction.ConstructGraph(self.nlp)
                graph = cg.construct_graph(text)
            elif self.spacy:
                cg = graph_construction_chunks.ConstructGraph(self.nlp)
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

            all_uniques[file] = unique_key

            if self.n_grams == 1:
                inunique = {}
                freqtext = texttofreq(self.ps, text)
                for i in freqtext.keys():
                    if i in (unique_key.keys()):
                        inunique[i] = freqtext[i]
                finalunique = {k: v for k, v in sorted(inunique.items(), key=lambda item: item[1], reverse=True)}
            else:
                candidates = self.generate_candidates(text, list(unique_key.values()))
                finalunique = rank(self.ps, text, candidates)

            stemmed_keyphrases[file] = [[x] for x in finalunique]
            keyphrases[file] = [[x] for x in list(unique_key.values())]

        with open(os.path.join(self.params["path"], 'unigram.pkl'), 'wb') as f:
            pickle.dump(all_uniques, f)

        return keyphrases, stemmed_keyphrases


def main(config):
    dataset = config['data']
    data_dir = os.path.join(config['data_dir'], dataset)
    use_stanza = config['use_stanza']
    use_spacy = config['use_spacy']
    gpu = config['gpu']

    n_grams = config['n_grams']

    dataset_details = read_json(os.path.join(config['data_dir'], f'{dataset}.json'))

    if config['files'] == "all":
        files = dataset_details['files']
    else:
        files = ["test"]

    model = HyperRank(dataset_details, data_dir, n_grams, use_stanza, use_spacy, gpu)
    keyphrases, stemmed_keyphrases = model.run(files)

    with open(os.path.join(data_dir, "keyphrases.pkl"), "wb") as f:
        pickle.dump({"stemmed": stemmed_keyphrases, "keyphrases": keyphrases}, f)

    file_name = f'../log/{int(datetime.timestamp(datetime.now()))}.json'

    dump_to_file = {'config': config}

    if 'eval' in config:
        ground_truth = read_json(os.path.join(config['data_dir'], dataset_details['references']['test']))
        if config['eval'] == "all":
            for file in files[1:]:
                ground_truth = merge_dicts(ground_truth,
                                           read_json(
                                               os.path.join(config['data_dir'],
                                                            dataset_details['references'][file.split(".")[0]])))

        scores = {}
        for top in config['top']:
            scores[top] = evaluate_model(stemmed_keyphrases, ground_truth, top, True)

        dump_to_file['score'] = scores

    with open(file_name, 'w') as f:
        json.dump(dump_to_file, f)


if __name__ == '__main__':
    dataset = [os.path.basename(x).split('.')[0] for x in glob('../data/*.json')]
    files = ["test", "all"]
    n_grams = [1, -1]
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
    parser.add_argument("--use-spacy",
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
    parser.add_argument("--n-grams",
                        type=int,
                        choices=n_grams,
                        default=1,
                        help="Generate n grams")
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
