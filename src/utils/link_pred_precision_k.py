import numpy as np
import networkx as nx


# class Dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)


def check_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind / data.N
            y = ind % data.N
            count += 1
            if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret


def check_link_prediction(embedding, train_graph: nx.Graph, origin_graph: nx.Graph, check_index):
    def get_precisionK(embedding, train_graph: nx.Graph, origin_graph: nx.Graph, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        sortedInd = sortedInd[::-1]
        cur = 0
        count = 0
        precisionK = []
        N = train_graph.number_of_nodes()
        for ind in sortedInd:
            x = ind / N
            y = ind % N
            if x == y or train_graph.has_edge(x, y):
                continue
            count += 1
            if origin_graph.has_edge(x, y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, train_graph, origin_graph, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret
