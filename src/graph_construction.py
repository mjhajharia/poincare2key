import string

import networkx as nx


class ConstructGraph:
    def __init__(self, nlp):
        self.nlp = nlp

    @staticmethod
    def __load_punctuation():
        return [x for x in string.punctuation]

    @staticmethod
    def __load_stopwords():
        stopwords = []
        with open("../data/stopwords_en_yake.txt", 'r', encoding="utf8") as File:
            for line in File.readlines():
                stopwords.append(str(line).strip())

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

    def construct_graph(self, text_data, graph=True):
        doc = self.nlp(text_data)

        edges, id2word = self.__get_edges_mapping(doc)

        complete_graph = self.__get_complete_graph(edges)

        combined_graph = self.__join_sentence_graphs(complete_graph, id2word)
        if graph:
            return nx.DiGraph(combined_graph)
        else:
            return combined_graph
