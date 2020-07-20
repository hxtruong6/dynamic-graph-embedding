import networkx as nx
from time import time
import matplotlib.pyplot as plt

from sklearn.manifold import LocallyLinearEmbedding
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import scipy.sparse.linalg as lg
import seaborn as sns


class LocallyLinearEmbedding():
    def __init__(self, dimension=2):
        '''

        :param dimension: dimension of the embedding
        '''
        self.dimension = dimension

    def learn_embedding(self, graph):
        graph = graph.to_undirected()

        start_time = time()

        A = nx.to_scipy_sparse_matrix(graph)
        print(A.todense())
        normalize(A, norm='l1', axis=1, copy=False)
        I_n = sp.eye(graph.number_of_nodes())
        I_min_A = I_n - A
        print(I_min_A)
        u, s, vt = lg.svds(I_min_A, k=self.dimension + 1, which='SM')

        finish_time = time()
        self._X = vt.T
        self._X = self._X[:, 1:]
        return self._X, (finish_time - start_time)

    def get_embedding(self):
        return self._X


if __name__ == '__main__':
    print("Locally Linear Embedding method")
    graph = nx.karate_club_graph()
    #     Draw graph
    nx.draw_networkx(graph, with_labels=True)
    plt.show()

    embedding = LocallyLinearEmbedding()
    graph_embedding, execute_time = embedding.learn_embedding(graph)

    print(graph_embedding)

    sns.scatterplot(data=graph_embedding)
    # nx.draw_networkx(nx.from_numpy_matrix(graph_embedding, parallel_edges=True), with_labels=True)
    plt.show()
