from time import time
import networkx as nx
from gensim.test.utils import get_tmpfile

from utils.skip_gram import SkipGram
from utils.visualize import plot_embeddings_with_labels
from utils.graph_walker_ import GraphWalker


class DeepWalk():
    def __init__(self, graph, walks_per_vertex, walk_length):
        """

        :param graph: graph input
        :param walks_per_vertex: (gamma param in paper) the number of looping through over all of nodes in graph
        :param walk_length: length of each walk path start node u
        """
        self.graph = graph
        self.graph_walker = GraphWalker(self.graph)
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
    test_num = 1

    # Test 1
    if test_num == 1:
        G = nx.read_edgelist('../data/Wiki_edgelist.txt',
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

        model = DeepWalk(G, walks_per_vertex=20, walk_length=10)
        model.train(embedding_size=128, window_size=5)
        embeddings = model.get_embedding()
        plot_embeddings_with_labels(G, embeddings, path_file="../data/Wiki_category.txt")

    # Test 2
    if test_num == 2:
        G = nx.karate_club_graph()

        model = DeepWalk(G, walks_per_vertex=20, walk_length=10)
        model.train(embedding_size=128, window_size=5)
        embeddings = model.get_embedding()
        plot_embeddings_with_labels(G, embeddings)
