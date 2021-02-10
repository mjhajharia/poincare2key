import networkx as nx


class ConstructGraph:
    def __init__(self, nlp):
        self.nlp = nlp

    def construct_graph(self, text_data, graph=True):
        doc = self.nlp(text_data)

        g = nx.DiGraph(n=0)
        for chunk in doc.noun_chunks:
            g.add_node(chunk.text)
            g.add_edge(chunk.text, chunk.root.head.text)
            if chunk.root.head.head:
                g.add_edge(chunk.root.head.text, chunk.root.head.head.text)
                if chunk.root.head.head.head:
                    g.add_edge(chunk.root.head.head.text, chunk.root.head.head.head.text)
                    if chunk.root.head.head.head.head:
                        g.add_edge(chunk.root.head.head.head.text, chunk.root.head.head.head.head.text)
                        if chunk.root.head.head.head.head.head:
                            g.add_edge(chunk.root.head.head.head.head.text, chunk.root.head.head.head.head.head.text)

        if graph:
            return g
        else:
            return list(g.edges())
