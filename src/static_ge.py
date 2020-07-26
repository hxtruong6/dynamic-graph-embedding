import tensorflow as tf
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.data_preprocessing.graph_dataset import GraphDataset
from src.data_preprocessing.graph_preprocessing import next_datasets, get_graph_from_file
from src.utils.autoencoder import Autoencoder, TAutoencoder
from src.utils.graph_util import draw_graph, print_graph_stats
from src.utils.model_utils import save_custom_model
from src.utils.visualize import plot_losses, plot_embeddings_with_labels, plot_embedding


class StaticGE(object):
    def __init__(self, G, embedding_dim=None, hidden_dims=[], model: Autoencoder = None, alpha=0.01, beta=2, nu1=0.001,
                 nu2=0.001):
        super(StaticGE, self).__init__()
        self.G = nx.Graph(G)
        self.alpha = alpha
        self.beta = beta

        if model is None:
            self.embedding_dim = embedding_dim
            self.hidden_dims = hidden_dims
            self.input_dim = self.G.number_of_nodes()
            self.model = Autoencoder(
                input_dim=self.input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dims=self.hidden_dims,
                v1=nu1,
                v2=nu2
            )
        else:
            self.model = model

        self.A, self.L = self.create_A_L_matrix()

    def compute_loss(self, model, inputs, alpha=0.01, beta=2):
        '''

        :param model:
        :param inputs:
        :param alpha:
        :param beta:
        :return:
        '''

        def loss_1st(Y, L):
            # 2 * tr(Y^T * L * Y)
            return 2 * tf.linalg.trace(
                tf.linalg.matmul(tf.linalg.matmul(tf.transpose(Y), L), Y)
            )

        def loss_2nd(X_hat, X, beta):
            B = np.ones_like(X)
            B[X != 0] = beta
            return tf.reduce_sum(tf.pow((X_hat - X) * B, 2))

        X, L = inputs
        X_hat, Y = model(X)

        # batch_size = X.shape[0]
        # TODO: check if divide batch_size

        loss_1 = loss_1st(Y, L)
        loss_2 = loss_2nd(X_hat, X, beta)
        return loss_2 + alpha * loss_1

    def train(self, batch_size=1, epochs=1, learning_rate=0.003, skip_print=5, save_model_point=50,
              model_folder_path=None):
        def train_step(loss, model, opt, inputs, alpha, beta):
            with tf.GradientTape() as tape:
                gradients = tape.gradient(
                    loss(model, inputs, alpha, beta),
                    model.trainable_variables
                )
            gradient_variables = zip(gradients, model.trainable_variables)
            opt.apply_gradients(gradient_variables)

        # ---------
        writer = tf.summary.create_file_writer('tmp')

        graph_embedding_list = []
        losses = []

        # datasets = get_transform(self.tf_dataset, batch_size=1, prefetch_times=1, shuffle=False)
        # tf.keras.backend.clear_session()
        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        with writer.as_default():
            with tf.summary.record_if(True):
                for epoch in range(epochs):
                    epoch_loss = []
                    for step, batch_inp in next_datasets(self.A, self.L, batch_size=batch_size):
                        train_step(self.compute_loss, self.model, opt, batch_inp, alpha=self.alpha,
                                   beta=self.beta)
                        loss_values = self.compute_loss(self.model, batch_inp, alpha=self.alpha,
                                                        beta=self.beta)
                        epoch_loss.append(loss_values)
                    # tf.summary.scalar('loss', loss_values, step=epoch)
                    # embedding = self.get_embedding()
                    mean_epoch_loss = np.mean(epoch_loss)
                    if epoch == 0 or (epoch + 1) % skip_print == 0:
                        print(f"\tEpoch {epoch + 1}: Loss = {mean_epoch_loss}")
                    losses.append(mean_epoch_loss)
                    if model_folder_path is not None \
                            and save_model_point is not None \
                            and (epoch + 1) % save_model_point == 0:
                        save_custom_model(model=self.model, model_folder_path=model_folder_path, checkpoint=epoch + 1)

                plot_losses(losses, title="Train GE", x_label="Epoch", y_label="Loss value")
                print(f"Loss = {losses[-1]}")

    def create_A_L_matrix(self):
        A = nx.to_scipy_sparse_matrix(self.G, format='csr').astype(np.float32)
        D = sparse.diags(A.sum(axis=1).flatten().tolist()[0]).astype(np.float32)
        L = D - A
        return A, L

    def get_embedding(self, inputs=None):
        if inputs is None:
            inputs = nx.adj_matrix(self.G).todense()

        return self.model.get_embedding(inputs).numpy()

    def get_model(self):
        return self.model

    # def save_model(self, filepath=None):
    #     if filepath is None:
    #         raise ValueError("filepath must be not None")
    #     self.model._set_inputs(nx.adj_matrix(self.G).todense()[:1])
    #     self.model.save(filepath="./")
    #     # tf.keras.models.save_model(self.model, filepath=filepath)
    #
    # def load_model(self, filepath=None):
    #     if filepath is None:
    #         raise ValueError("filepath must be not None")
    #     self.model = tf.keras.models.load_model(filepath)


class TStaticGE(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def train(self, batch_size=1, shuffle=False, epochs=1, learning_rate=1e-3, skip_print=5, save_model_point=50,
              model_folder_path=None):
        # TODO: set seed through parameter
        torch.manual_seed(6)
        graph_dataset = GraphDataset(A=self.A, L=self.L)
        dataloader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=shuffle)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # mean-squared error loss
        criterion = nn.MSELoss()
        print(self.model)
        for epoch in range(epochs):
            for data in dataloader:
                A, L = data
                inp = Variable(A).to(self.device)
                # ===================forward=====================
                x_hat, y = self.model(inp)
                loss = criterion(x_hat, inp)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
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


if __name__ == "__main__":
    G = get_graph_from_file(filename="../data/email-eu/email-Eu-core.txt")
    # G = nx.gnm_random_graph(n=70, m=150, seed=6)
    print_graph_stats(G)
    draw_graph(G)

    ge = TStaticGE(G=G, embedding_dim=4, hidden_dims=[32, 16])
    ge.train(batch_size=5, epochs=10, skip_print=10, learning_rate=0.001)
    embeddings = ge.get_embedding()
    # classify_embeddings_evaluate(embeddings, label_file="../data/email-eu/email-Eu-core-department-labels.txt")
    plot_embeddings_with_labels(G, embeddings=embeddings,
                                path_file="../data/email-eu/email-Eu-core-department-labels.txt")
    plot_embedding(embeddings=embeddings)
