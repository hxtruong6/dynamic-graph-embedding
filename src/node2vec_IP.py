from time import time

import networkx as nx
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from joblib import cpu_count

# TODO: in processing
from src.utils.graph_walker_ import GraphWalker
from src.utils.skip_gram import SkipGram
from src.utils.visualize import plot_embeddings_with_labels


class Node2Vec:
    def __init__(self, graph, walks_per_vertex, walk_length, p, q):
        """

        :param graph: graph input
        :param walks_per_vertex: (gamma param in paper) the number of looping through over all of nodes in graph
        :param walk_length: length of each walk path start node u
        :param p: return parameter
        :param q: in-out parameter
        """
        self.graph = graph
        self.p = p
        self.q = q

        self.graph_walker = GraphWalker(self.graph)
        self.graph_walker.preprocess_transition_probs()
        self.walks_corpus = self.graph_walker.build_walk_corpus(walks_per_vertex, walk_length)

    def train(self, embedding_size, window_size):
        print("Start training...")
        start_time = time()
        self.model = SkipGram(self.walks_corpus, embedding_size, window_size)

        finish_time = time()
        print("Done! Training time: ", finish_time - start_time)

    def get_embedding(self):
        self.embedding = {}
        for node in list(self.graph.nodes()):
            self.embedding[str(node)] = self.model.wv[str(node)]

        return self.embedding

    def save_model(self, path=None):
        if path is None:
            path = get_tmpfile("node_vectors.kv")
        self.model.wv.save(path)


if __name__ == "__main__":
    # Change this to run test
    test_num = 2

    # Test 1
    if test_num == 1:
        G = nx.read_edgelist('../data/Wiki_edgelist.txt',
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

        model = Node2Vec(G, walks_per_vertex=20, walk_length=10, p=0.25, q=4)
        model.train(embedding_size=128, window_size=5)
        embeddings = model.get_embedding()
        plot_embeddings_with_labels(G, embeddings, path_file="../data/Wiki_category.txt")

    # Test 2
    if test_num == 2:
        G = nx.karate_club_graph()

        model = Node2Vec(G, walks_per_vertex=20, walk_length=10, p=0.25, q=1.5)
        model.train(embedding_size=128, window_size=5)
        embeddings = model.get_embedding()
        plot_embeddings_with_labels(G, embeddings)
