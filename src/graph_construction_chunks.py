import networkx as nx


class ConstructGraph:
    def __init__(self, nlp):
        self.nlp = nlp

    def construct_graph(self, text_data, graph=True):
        text = text_data.split('.')
        g = nx.DiGraph(n=0)
        for x in text:
            doc = self.nlp(x)
            for chunk in doc.noun_chunks:
                g.add_edge(chunk.text.lower(),chunk.root.head.text.lower())        
                x = chunk.root
                while x.head.text.lower()!=x.head.head.text.lower():
                    g.add_edge(x.head.text.lower(), x.head.head.text.lower())
                    x = x.head
        if graph:
            return g
        else:
            return list(g.edges())
