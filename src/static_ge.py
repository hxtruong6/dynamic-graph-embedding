import tensorflow as tf
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import torch
from torch.autograd import Variable

from src.data_preprocessing.graph_preprocessing import next_datasets
from src.utils.autoencoder import TAutoencoder
from src.utils.graph_util import draw_graph, print_graph_stats
from src.utils.visualize import plot_reconstruct_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TStaticGE(object):

    def __init__(self, G, embedding_dim=None, hidden_dims=[], model: TAutoencoder = None, alpha=0.01, beta=2, nu1=0.001,
                 nu2=0.001):
        super(TStaticGE, self).__init__()
        self.G = nx.Graph(G)
        self.alpha = alpha
        self.beta = beta
        self.l1 = nu1
        self.l2 = nu2

        if model is None:
            self.embedding_dim = embedding_dim
            self.hidden_dims = hidden_dims
            self.input_dim = self.G.number_of_nodes()
            self.model = TAutoencoder(
                input_dim=self.input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dims=self.hidden_dims,
                l1=nu1,
                l2=nu2
            )
        else:
            self.model = model

        self.A: sparse.csr_matrix
        self.L: sparse.csr_matrix
        self.A, self.L = self._create_A_L_matrix()

    def _create_A_L_matrix(self):
        A = nx.to_scipy_sparse_matrix(self.G, format='csr').astype(np.float32)
        D = sparse.diags(A.sum(axis=1).flatten().tolist()[0]).astype(np.float32)
        L = D - A
        return A, L

    def _compute_loss(self, x, x_hat, y, L):
        def loss_1st(Y, L):
            Y_ = torch.matmul(torch.transpose(Y, 0, 1), L)
            return 2 * torch.trace(torch.matmul(Y_, Y))

        def loss_2nd(X_hat, X, beta):
            B = np.ones_like(X)
            B[X != 0] = beta
            return torch.sum(torch.square((X_hat - X) * torch.tensor(B)))

        batch_size = x.shape[0]
        # TODO: check if divide batch_size
        loss_1 = loss_1st(y, L)
        loss_2 = loss_2nd(x_hat, x, self.beta) / batch_size
        loss = loss_2 + self.alpha * loss_1
        return loss

    def train(self, batch_size=1, epochs=1, learning_rate=1e-3, skip_print=5):
        # TODO: set seed through parameter
        torch.manual_seed(6)
        # graph_dataset = GraphDataset(A=self.A, L=self.L)
        # dataloader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False, sampler=)

        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # for data in dataloader:
            loss = None
            for step, batch_inp in next_datasets(self.A, self.L, batch_size=batch_size):
                A, L = batch_inp
                A = torch.tensor(A)
                L = torch.tensor(L)

                x = Variable(A).to(device)
                # ===================forward=====================
                x_hat, y = self.model(x)
                loss = self._compute_loss(x, x_hat, y, L)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            if (epoch + 1) % skip_print == 0 or epoch == epochs - 1 or epoch == 0:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss))

    def get_embedding(self, x=None):
        '''

        :param x: graph input. Must have same dimension with the original graph
        :return:
        '''
        if x is None:
            x = self.A.todense()
        # Convert to tensor for pytorch
        x = torch.tensor(x)
        embedding = self.model.get_embedding(x=x)
        return embedding

    def get_reconstruction(self):
        x = torch.tensor(self.A.todense())
        return self.model.get_reconstruction(x=x)

    def get_model(self):
        return self.model


if __name__ == "__main__":
    # G = get_graph_from_file(filename="../data/email-eu/email-Eu-core.txt")
    G = nx.gnm_random_graph(n=11, m=15, seed=6)
    print_graph_stats(G)
    draw_graph(G)
    pos = nx.spring_layout(G, seed=6)

    ge = TStaticGE(G=G, embedding_dim=2, hidden_dims=[8, 4])
    # ge = StaticGE(G=G, embedding_dim=2, hidden_dims=[8, 4])
    ge.train(batch_size=3, epochs=2000, skip_print=100, learning_rate=0.003)
    embeddings = ge.get_embedding()
    reconstructed_graph = ge.get_reconstruction()
    # classify_embeddings_evaluate(embeddings, label_file="../data/email-eu/email-Eu-core-department-labels.txt")
    # plot_embeddings_with_labels(G, embeddings=embeddings,
    #                             path_file="../data/email-eu/email-Eu-core-department-labels.txt")

    # plot_embedding(embeddings=embeddings)
    plot_reconstruct_graph(reconstructed_graph=reconstructed_graph, pos=pos, threshold=0.6)
