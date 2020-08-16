import pickle
from os.path import join
from time import time
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import torch
from node2vec import Node2Vec
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.data_preprocessing.graph_dataset import GraphDataset
from src.data_preprocessing.graph_preprocessing import next_datasets, get_graph_from_file, handle_graph_mini_batch
from src.utils.autoencoder import TAutoencoder
from src.utils.checkpoint_config import CheckpointConfig
from src.utils.graph_util import draw_graph, print_graph_stats
from src.utils.link_pred_precision_k import check_link_prediction
from src.utils.link_prediction import preprocessing_graph_for_link_prediction, run_link_pred_evaluate
from src.utils.model_utils import save_custom_model
from src.utils.visualize import plot_reconstruct_graph, plot_embeddings_with_labels, plot_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TStaticGE(object):
    def __init__(self, G: nx.Graph, embedding_dim=None, hidden_dims=None, model: TAutoencoder = None,
                 alpha=0.2, beta=8, l1=0., l2=1e-5, activation='relu'):
        super(TStaticGE, self).__init__()
        if hidden_dims is None:
            hidden_dims = []
        self.G = G
        # TODO: set alpha beta in If statement
        self.alpha = alpha
        self.beta = beta

        if model is None:
            self.embedding_dim = embedding_dim
            self.hidden_dims = hidden_dims
            self.input_dim = self.G.number_of_nodes()
            self.l1 = l1
            self.l2 = l2
            self.model = TAutoencoder(
                input_dim=self.input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dims=self.hidden_dims,
                l1=l1,
                l2=l2,
                activation=activation
            )
        else:
            self.model = model
            config_layer = self.model.get_config_layer()
            self.embedding_dim = config_layer['embedding_dim']
            self.hidden_dims = config_layer['hidden_dims']
            self.input_dim = config_layer['input_dim']
            self.l1 = config_layer['l1']
            self.l2 = config_layer['l2']

        self.A: sparse.csr_matrix
        self.L: sparse.csr_matrix
        self.A, self.L = self._create_A_L_matrix()
        self.embedding = None

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
            B = torch.ones_like(X).to(device)
            B[X != 0] = beta
            return torch.sum(torch.square((X_hat - X) * B))

        batch_size = x.shape[0]
        # TODO: check if divide batch_size
        loss_1 = loss_1st(y, L)
        loss_2 = loss_2nd(x_hat, x, self.beta)
        loss = loss_2 + self.alpha * loss_1
        return loss

    def train(self, batch_size=None, epochs=1, learning_rate=1e-6, skip_print=1, ck_config: CheckpointConfig = None,
              early_stop=None, threshold_loss=1e-4, plot_loss=False, shuffle=False):
        # TODO: set seed through parameter
        torch.manual_seed(6)
        # graph_dataset = GraphDataset(A=self.A, L=self.L, batch_size=batch_size)
        # dataloader = DataLoader(graph_dataset)
        graph_dataset = GraphDataset(A=self.A, L=self.L)
        if batch_size is None:
            batch_size = self.input_dim
        dataloader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.l2)

        min_loss = 1e6
        count_epoch_no_improves = 0
        train_losses = []
        is_stop_train = False
        for epoch in range(epochs):
            # for data in dataloader:
            t1 = time()
            epoch_loss = 0
            if is_stop_train:
                break
            for step, batch_inp in enumerate(dataloader):
                A, L = handle_graph_mini_batch(batch_inp)

                # Trick here. TODO: check why A is (1,batch_size,number_nodes)
                # x = Variable(A[0]).to(device)
                # L = L[0].to(device)
                x = Variable(A).to(device)
                L = L.to(device)
                # ===================forward=====================
                optimizer.zero_grad()

                x_hat, y = self.model(x)
                loss = self._compute_loss(x, x_hat, y, L)
                epoch_loss += loss

                if loss < 0:
                    is_stop_train = True
                    print("Stopping training due to negative loss.")
                    break

                # ===================backward====================
                loss.backward()
                optimizer.step()

                del x, L
                # ===================log========================
            train_losses.append(round(float(epoch_loss), 4))
            if (epoch + 1) % skip_print == 0 or epoch == epochs - 1 or epoch == 0:
                print(
                    'Epoch [{}/{}] \t\tloss:{:.4f} \t\ttime:{:.2f}s'.format(epoch + 1, epochs, epoch_loss, time() - t1))

            if ck_config is not None and ck_config.NumberSaved == epoch:
                save_custom_model(model=self.model, filepath=join(ck_config.FolderPath, f"graph_{ck_config.Index}"))

            if epoch_loss < min_loss - threshold_loss:
                count_epoch_no_improves = 0
                min_loss = epoch_loss
            else:
                count_epoch_no_improves += 1

            if early_stop is not None and count_epoch_no_improves == early_stop:
                print('Early stopping!\t Epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, epoch_loss))
                break

            torch.cuda.empty_cache()

        if plot_loss:
            plot_losses(losses=train_losses, x_label="epoch", y_label="loss")

    def get_embedding(self, x=None):
        '''

        :param x: graph input. Must have same dimension with the original graph
        :return:
        '''
        if x is None:
            x = self.A.todense()
        # Convert to tensor for pytorch
        x = torch.tensor(x).to(device)

        with torch.no_grad():
            embedding = self.model.to(device).get_embedding(x=x)
        torch.cuda.empty_cache()
        self.embedding = embedding
        return embedding

    def get_reconstruction(self, x=None):
        if x is None:
            x = self.A.todense()
        x = torch.tensor(x).to(device)
        with torch.no_grad():
            reconstruction = self.model.to(device).get_reconstruction(x=x)
        torch.cuda.empty_cache()
        return reconstruction

    def get_model(self):
        return self.model.cpu()

    def save_embedding(self, filepath):
        if self.embedding is None:
            self.embedding = self.get_embedding()
        with open(filepath, 'wb') as fp:
            pickle.dump(self.embedding, fp)

    def load_embedding(self, filepath):
        with open(filepath, 'rb') as fp:
            self.embedding = pickle.load(fp)
        return self.embedding


if __name__ == "__main__":
    # G = get_graph_from_file(filename="../data/email-eu/email-Eu-core.txt")
    G = nx.gnm_random_graph(n=40, m=100, seed=6)
    print_graph_stats(G)
    pos = nx.spring_layout(G, seed=6)
    draw_graph(G, limit_node=50, pos=pos)
    # print(G.edges)

    embedding_dim = 4

    g_hidden_df, hidden_G = preprocessing_graph_for_link_prediction(
        G=G,
        drop_node_percent=0.2,
        edge_rate=0.1
    )

    ge = TStaticGE(G=hidden_G, embedding_dim=embedding_dim, hidden_dims=[16, 8], l2=1e-5, alpha=0.2, beta=10,
                   activation='relu')
    start_time = time()
    ck_point = CheckpointConfig(number_saved=2, folder_path="../data")
    ge.train(batch_size=None, epochs=10000, skip_print=500,
             learning_rate=5e-5, early_stop=200, threshold_loss=1e-4,
             plot_loss=True, shuffle=True, ck_config=ck_point
             )

    # ge.train(batch_size=128, epochs=10000, skip_print=500, learning_rate=0.001, early_stop=200, threshold_loss=1e-4)
    print(f"Finished in {round(time() - start_time, 2)}s")
    embedding = ge.get_embedding()
    reconstructed_graph = ge.get_reconstruction()

    print(embedding[:3])
    print(nx.adjacency_matrix(G).todense()[:3])
    print(reconstructed_graph[:3])
    link_pred_prec = check_link_prediction(embedding, train_graph=hidden_G, origin_graph=G, k_query=[2, 10, 20])
    print("Precision@K: ", link_pred_prec)

    run_link_pred_evaluate(
        graph_df=g_hidden_df,
        embeddings=embedding,
        num_boost_round=20000
    )
    # print(reconstructed_graph)
    # classify_embeddings_evaluate(embeddings, label_file="../data/email-eu/email-Eu-core-department-labels.txt")
    # plot_embeddings_with_labels(G, embeddings=embeddings,
    #                             path_file="../data/email-eu/email-Eu-core-department-labels.txt",
    #                             save_path="../images/Email-static-ge")

    # save_custom_model(ge.get_model(), filepath="../models/email-eu/email-eu")

    # plot_embedding(embeddings=embeddings)
    plot_reconstruct_graph(reconstructed_graph=reconstructed_graph, pos=pos, threshold=0.6)

    # print("========= Node2vec ==========")
    # node2vec = Node2Vec(graph=hidden_G,
    #                     dimensions=embedding_dim,
    #                     walk_length=80,
    #                     num_walks=20,
    #                     workers=2)  # Use temp_folder for big graphs
    # node2vec_model = node2vec.fit()
    # embedding = [node2vec_model[str(u)] for u in sorted(hidden_G.nodes)]
    # embedding = np.array(embedding)
    # print(embedding[:3])
    #
    # link_pred_prec = check_link_prediction(embedding, train_graph=hidden_G, origin_graph=G, check_index=[2, 10, 20])
    # print("Precision@K: ", link_pred_prec)
    # run_link_pred_evaluate(
    #     graph_df=g_hidden_df,
    #     embeddings=embedding,
    #     num_boost_round=20000
    # )
