import string

import networkx as nx
import stanza
from pymagnitude import *

from utils import read_json

from nltk.stem import PorterStemmer

ps = PorterStemmer()

data_json_path = {
    'marujo': 'mj.json',
    'hulth': 'ah.json',
    'sem-eval': 'se.json'
}


class LoadData:
    def __init__(self, config):
        self.config = config
        self.data_path = data_json_path[self.config['dataset']]
        self.data_dir = self.config['data_dir']

        self.data = read_json(os.path.join(self.data_dir, self.data_path))
        self.nlp = self.__load_stanza()

        glove = Magnitude(os.path.join(self.data_dir, 'glove.6B.50d.magnitude'))
        pos_vectors = FeaturizerMagnitude(100, namespace="PartsOfSpeech")
        dependency_vectors = FeaturizerMagnitude(
            100, namespace="SyntaxDependencies")
        self.vectors = Magnitude(glove, pos_vectors, dependency_vectors)

    @staticmethod
    def __load_stanza():
        preprocessors = 'tokenize,mwt,pos,lemma,depparse'

        return stanza.Pipeline(lang='en', processors=preprocessors)

    @staticmethod
    def __load_punctuation():
        return [x for x in string.punctuation]

    @staticmethod
    def __load_stopwords():
        stopwords = []
        with open(r"stopwords_en_yake.txt", 'r', encoding="utf8") as File:
            for line in File.readlines():
                stopwords.append(str(line)[:-1])

        return stopwords

    @staticmethod
    def __get_complete_graph(all_edges):
        complete_graph = []
        for edges in all_edges:
            for edge in edges:
                complete_graph.append(edge)
        return complete_graph

    def __join_sentence_graphs(self, complete_graph, all_words):

        stop_words = self.__load_stopwords()
        punctuation = self.__load_punctuation()

        all_words = list(all_words)

        for i in range(len(all_words) - 1):
            for j in range(i + 1, len(all_words)):
                for v1 in all_words[i]:
                    if v1 in stop_words + punctuation or len(v1) < 4:
                        continue
                    for v2 in all_words[j]:
                        if v2 in stop_words + punctuation or len(v2) < 4:
                            continue
                        if v1.lower() == v2.lower():
                            complete_graph.append((str(v1), str(v2)))
        return complete_graph

    def __get_edges_mapping(self, doc):
        all_edges = []

        punctuation = self.__load_punctuation()
        all_words = set()
        for i, sent in enumerate(doc.sentences):
            edges = []
            for word in sent.words:
                if word.text not in punctuation:
                    all_words.add(ps.stem(word.text))

                    if word.head > 0:
                        edges.append((
                            f'{ps.stem(sent.words[word.head - 1].text if word.head > 0 else "root").lower()}',
                            f'{ps.stem(word.text).lower()}'))

            all_edges.append(edges)

        return all_edges, all_words

    def construct_graph(self, text_data):
        doc = self.nlp(text_data)

        edges, all_words = self.__get_edges_mapping(doc)

        complete_graph = self.__get_complete_graph(edges)

        combined_graph = self.__join_sentence_graphs(complete_graph, all_words)

        return nx.DiGraph(combined_graph)

    @staticmethod
    def vector_preprocess(doc):
        lemmas = []
        for i, sent in enumerate(doc.sentences):
            for word in sent.words:
                lemmas.append((word.lemma.lower(),
                               word.pos,
                               word.xpos,
                               f'{word.text}.{i}.{word.id}'))
        return lemmas

    def vectorization(self, text_data):
        doc = self.nlp(text_data)

        words = self.vector_preprocess(doc)

        word_vector_map = {}

        for word in words:
            try:
                # vector, (all other lemma except id)
                word_vector_map[word[-1]] = [self.vectors.query(word[:-1]), word[:-1]]
            except:
                word_vector_map[word[-1]] = [np.zeros(self.vectors.dim), word[:-1]]

        return word_vector_map
