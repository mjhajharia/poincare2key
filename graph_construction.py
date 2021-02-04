import string

import networkx as nx
import stanza
# from pymagnitude import *

from utils import read_json
import os

data_json_path = {
    'marujo': 'mj.json',
    'hulth': 'ah.json',
    'sem-eval': 'se.json',
    'semeval2017':'semeval2017.json'
}


class LoadData:
    def __init__(self, config):
        self.config = config
        self.data_path = data_json_path[self.config['dataset']]
        self.data_dir = self.config['data_dir']

        self.data = read_json(os.path.join(self.data_dir, self.data_path))
        self.nlp = self.__load_stanza()

        # glove = Magnitude(os.path.join(self.data_dir, 'glove.6B.50d.magnitude'))
        # pos_vectors = FeaturizerMagnitude(100, namespace="PartsOfSpeech")
        # dependency_vectors = FeaturizerMagnitude(100, namespace="SyntaxDependencies")
        # self.vectors = Magnitude(glove, pos_vectors, dependency_vectors)

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

    def __join_sentence_graphs(self, complete_graph, id2word):

        stop_words = self.__load_stopwords()
        punctuation = self.__load_punctuation()

        for i in range(len(id2word) - 1):
            for j in range(i + 1, len(id2word)):
                for k1, v1 in id2word[i].items():
                    if v1 in stop_words + punctuation or len(v1) < 4:
                        continue
                    for k2, v2 in id2word[j].items():
                        if v2 in stop_words + punctuation or len(v2) < 4:
                            continue
                        if v1.lower() == v2.lower():
                            complete_graph.append((str(k1), str(k2)))
        return complete_graph

    def __get_edges_mapping(self, doc):
        all_edges = []
        sentence_id2word = []

        punctuation = self.__load_punctuation()

        for i, sent in enumerate(doc.sentences):
            id2word = {}
            edges = []
            for word in sent.words:
                if word.text not in punctuation:
                    id2word[word.text + '.' +
                            str(i) + '.' + str(word.id)] = word.lemma

                    if word.head > 0:
                        edges.append((
                            f'{sent.words[word.head - 1].text if word.head > 0 else "root"}.{i}.{word.head if word.head > 0 else "root"}',
                            f'{word.text}.{i}.{word.id}'))

            all_edges.append(edges)
            sentence_id2word.append(id2word)

        return all_edges, sentence_id2word

    def construct_graph(self, text_data):
        doc = self.nlp(text_data)

        edges, id2word = self.__get_edges_mapping(doc)

        complete_graph = self.__get_complete_graph(edges)

        combined_graph = self.__join_sentence_graphs(complete_graph, id2word)

        return nx.DiGraph(combined_graph)

    @staticmethod
    def vector_preprocess(tokens, doc):
        lemmas = []
        for i, sent in enumerate(doc.sentences):
            for word in sent.words:
                    for i in tokens:
                        if i==word.lower():
                            lemmas.append((word.lemma.lower(),
                                           word.pos,
                                           word.xpos,
                                           f'{word.text}.{i}.{word.id}'))
        return lemmas
    
    def vectorization(self, text_data, tokens):
        doc = self.nlp(text_data)
    
        words = self.vector_preprocess(tokens, doc)
    
        word_vector_map = {}
    
        for word in words:
            try:
                # vector, (all other lemma except id)
                word_vector_map[word[-1]] = [self.vectors.query(word[:-1]), word[:-1]]
            except:
                word_vector_map[word[-1]] = [np.zeros(self.vectors.dim), word[:-1]]
    
        return word_vector_map
