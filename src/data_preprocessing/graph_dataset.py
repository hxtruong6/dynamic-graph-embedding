from __future__ import print_function, division
import os

import networkx as nx
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from torch.utils.data import Dataset, DataLoader

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph, next_datasets
from src.utils.graph_util import draw_graph


class GraphDataset(Dataset):
    """Dynamic graph dataset"""

    def __init__(self, A: sparse.csr_matrix, L: sparse.csr_matrix, batch_size=1):
        '''
        This is trick dataset for graph. I pass batch_size here so when training, DataLoader is always batch_size =1
        :param A:
        :param L:
        :param batch_size:
        '''
        # self.dts = []
        # dataset_size = A.shape[0]
        # steps_per_epoch = (dataset_size - 1) // batch_size + 1
        # for i in range(steps_per_epoch):
        #     index = np.arange(
        #         i * batch_size, min((i + 1) * batch_size, dataset_size))
        #     A_train = A[index, :].todense()
        #     L_train = L[index][:, index].todense()
        #
        #     A_train = torch.tensor(A_train)
        #     L_train = torch.tensor(L_train)
        #     batch_inp = [A_train, L_train]
        #     self.dts.append(batch_inp)
        self.A = A
        self.L = L
        self.size = A.get_shape()[0]

    def __len__(self):
        # return len(self.dts)
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [idx, self.A.getrow(idx).toarray()[0]]
        # return self.dts[idx]
        return sample


if __name__ == "__main__":
    G = nx.gnm_random_graph(n=7, m=15, seed=6)
    # draw_graph(G)

    A = nx.to_scipy_sparse_matrix(G, format='csr').astype(np.float32)
    D = sparse.diags(A.sum(axis=1).flatten().tolist()[0]).astype(np.float32)
    L = D - A

    print(A.todense())
    print(L.todense())

    graph_dataset = GraphDataset(A=A, L=L)
    dataloader = DataLoader(graph_dataset, batch_size=2, shuffle=False)

    for i_batch, sample_batched in enumerate(dataloader):
        print(f"[{i_batch}]")
        x_hat, y = sample_batched
        print(sample_batched)
        # print(sample_batched[0], sample_batched[1])
    #
    # for i, batch_inp in next_datasets(A, L, batch_size=1):
    #     # for i in range(len(graph_dataset)):
    #     print(f"--[{i}]")
    #     print(graph_dataset[i])
    #     print(batch_inp)
