from os import listdir
from os.path import isfile, join
from time import time
import networkx as nx
import numpy as np
import torch


def get_graph_from_file(filename):
    if filename is None:
        raise AssertionError("File name is None!")
    G = nx.read_edgelist(filename, comments="#", nodetype=int, data=(('weight', float),))
    return G


# def convert_sparse_matrix_to_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return SparseTensor(indices, coo.data, coo.shape)


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


def handle_graph_mini_batch(batch_inp):
    # print(batch_inp)
    mini_A = []
    for idx in batch_inp[0]:
        col = batch_inp[1][:, idx].clone()
        col = col.detach().numpy()
        mini_A.append(col)
    mini_A = np.transpose(np.array(mini_A))

    D = np.diag(np.sum(mini_A, axis=0))
    L = D - mini_A

    # A = torch.tensor(A)
    L = torch.tensor(L)
    return batch_inp[1], L


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


def convert_node2idx(g: nx.Graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in sorted(g.nodes):
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return node2idx


def get_dyn_graph_to_idx(graphs: []):
    node2idx = {}
    idx2node = []
    node_count = 0
    for node in graphs[0].nodes:
        node2idx[node] = node_count
        idx2node.append(node)
        node_count += 1

    for i in range(1, len(graphs)):
        add_nodes = list(set(graphs[i].nodes) - set(graphs[i - 1].nodes))
        for node in add_nodes:
            node2idx[node] = node_count
            idx2node.append(node)
            node_count += 1
    return node2idx, idx2node


def convert_graphs_to_idx(graphs):
    graphs_len = len(graphs)

    print("Start convert graph to index ...", end=" ")
    start_time = time()

    dfs = []
    for i in range(graphs_len):
        df = nx.to_pandas_edgelist(graphs[i])
        dfs.append(df)

    node2idx, idx2node = get_dyn_graph_to_idx(graphs)

    graphs2idx = []
    for i in range(graphs_len):
        dfs[i]['source'] = dfs[i]['source'].apply(lambda x: node2idx[x])
        dfs[i]['target'] = dfs[i]['target'].apply(lambda x: node2idx[x])
        g = nx.from_pandas_edgelist(dfs[i])
        graphs2idx.append(g)

    print(f"{round(time() - start_time, 2)}s")
    return graphs2idx, idx2node


def read_dynamic_graph(folder_path=None, limit=None, convert_to_idx=True):
    if folder_path is None:
        raise ValueError("folder_path must be provided.")

    graphs = []
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    files = sorted(files)

    for idx, file in enumerate(files):
        if limit is not None and idx == limit:
            break
        print(f"[{idx}]Reading {file}")
        G = get_graph_from_file(join(folder_path, file))
        graphs.append(G)

    if not convert_to_idx:
        return graphs, None

    return convert_graphs_to_idx(graphs)


if __name__ == "__main__":
    graphs = read_dynamic_graph(folder_path="../../data/as-733", limit=10)
    print(graphs)
    for g in graphs:
        print(nx.info(g))
