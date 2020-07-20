from os import listdir
from os.path import isfile, join
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow import SparseTensor


def get_graph_from_file(filename):
    if filename is None:
        raise AssertionError("File name is None!")
    G = nx.read_edgelist(filename, comments="#", nodetype=int, data=(('weight', float),))
    return G


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return SparseTensor(indices, coo.data, coo.shape)


def next_datasets(A, L, batch_size):
    '''

    :param A:
    :param L:
    :param batch_size:
    :return:
    '''
    dataset_size = A.shape[0]
    steps_per_epoch = (dataset_size - 1) // batch_size + 1
    i = 0
    while i < steps_per_epoch:
        index = np.arange(
            i * batch_size, min((i + 1) * batch_size, dataset_size))
        A_train = A[index, :].todense()
        L_train = L[index][:, index].todense()
        batch_inp = [A_train, L_train]

        yield i, batch_inp
        i += 1


def read_dynamic_graph(folder_path=None, limit=None):
    if folder_path is None:
        raise ValueError("folder_path must be provided.")

    graphs = []

    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for idx, file in enumerate(files):
        if limit is not None and idx == limit:
            break
        G = get_graph_from_file(join(folder_path, file))
        graphs.append(G)

    return graphs


def read_node_label(filename, skip_head=False):
    X = []
    Y = []
    with open(filename) as fi:
        if skip_head:
            fi.readline()
        for line in fi:
            vec = line.strip().split()
            X.append(int(vec[0]))
            Y.append(vec[1:])
    return X, Y


if __name__ == "__main__":
    graphs = read_dynamic_graph(folder_path="../../data/as-733", limit=10)
    print(graphs)
    for g in graphs:
        print(nx.info(g))
