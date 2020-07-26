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

    def __init__(self, A: sparse.csr_matrix, L: sparse.csr_matrix, transform=None):
        self.dts_len = A.shape[0]
        self.A = A
        self.L = L

    def __len__(self):
        return self.dts_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        A = self.A
        L = self.L
        # A_train = self.A[idx, :].todense().tolist()
        A_train = list(self.A.getrow(idx).todense().flat)
        L_train = list(self.L[idx][:, idx].todense().flat)

        A_train = torch.tensor(A_train)
        L_train = torch.tensor(L_train)

        sample = [A_train, L_train]

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
