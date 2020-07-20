from datetime import time

import networkx as nx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import History
from tensorflow_core.python.keras.regularizers import l1_l2
import numpy as np
import scipy.sparse as sp

from utils.evaluate import classify_embeddings_evaluate
from utils.graph_util import preprocess_graph


def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
    X = Input(shape=(node_size,))
    L = Input(shape=(None,))  # dummy layer

    fc = X
    for i in range(len(hidden_size)):
        # embedding layer
        if i == len(hidden_size) - 1:
            fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2), name='1st')(fc)
        else:  # normal hidden layer
            fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2))(fc)
    # assign embedding layer to Y
    Y = fc
    for i in reversed(range(len(hidden_size) - 1)):
        fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2))(fc)
    # assign last hidden layer has input size as Input layer. x = x_hat
    X_hat = Dense(node_size, 'relu', name='2nd')(fc)
    model = Model(inputs=[X, L], outputs=[X_hat, Y])
    emb = Model(inputs=X, outputs=Y)
    return model, emb


def l_1st(alpha):
    def loss_1st(y_true, y_pred):
        L = y_true
        Y = y_pred
        batch_size = tf.to_float(K.shape(L)[0])
        # 2* alpha * tr(Y.T*L*Y)
        return alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(Y), L), Y)) / batch_size


def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
        b_ = np.ones_like(y_true)
        b_[y_true != 0] = beta
        x = K.square((y_pred - y_true) * b_)
        t = K.sum(x, axis=-1, )
        return K.mean(t)


class SDNE:
    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5.0, nu1=1e-5, nu2=1e-4):
        '''

        :param graph:
        :param hidden_size:
        :param alpha:
        :param beta:
        :param nu1:
        :param n2:
        '''
        self.graph = nx.Graph(graph)
        self.idx2node, self.node2idx = preprocess_graph(self.graph)
        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2

        # Create matrix: Adj matrix, L matrix
        self.A, self.L = self._create_A_L(self.graph, self.node2idx)
        self.reset_model()
        self.inputs = [self.A, self.L]
        self._embedding = {}

    def reset_model(self, opt='adam'):
        self.model, self.emb_model = create_model(self.node_size, hidden_size=self.hidden_size, l1=self.nu1,
                                                  l2=self.nu2)
        # config model with optimizer and loss function.
        self.model.compile(optimizer=opt, loss="mse")
        self.get_embeddings()

    def evaluate(self):
        # TODO: batch_size should 2^n
        return self.model.evaluate(x=self.inputs, y=self.inputs, batch_size=self.node_size)

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1):
        print(self.model.summary())
        # return
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print('batch_size({0}) > node_size({1}), set batch_size = {1}'.format(batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit([self.A.todense(), self.L.todense()], [self.A.todense(), self.L.todense()],
                                  batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, verbose=verbose,
                                  shuffle=False, )

        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(
                        i * batch_size, min((i + 1) * batch_size, self.node_size)
                    )
                    A_train = self.A[index, :].todense()

                    L_mat_train = self.L[index][:, index].todense()
                    inp = [A_train, L_mat_train]
                    print("A_train: ", A_train)
                    print("L_mat_train: ", L_mat_train)
                    batch_losses = self.model.train_on_batch(inp, inp)
                    losses += batch_losses
                losses = losses / steps_per_epoch

                logs['loss'] = losses[0]
                logs['2nd_loss'] = losses[1]
                logs['1st_loss'] = losses[2]

                epoch_time = int(time.time() - start_time)
                hist.on_epoch_begin(epoch, logs)
                if verbose > 0:
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_lost: {3: .4f}'.format(
                    epoch_time, losses[0], losses[1], losses[2]
                ))
        return hist

    def get_embeddings(self):
        self._embeddings = {}
        embeddings = self.emb_model.predict(x=self.A.todense(), batch_size=self.node_size)
        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[look_back[i]] = embedding
        return self._embeddings

    def _create_A_L(self, graph, node2idx):
        graph = nx.Graph(graph)
        node_size = graph.number_of_nodes()
        A_data = []
        A_row_index = []
        A_col_index = []

        for edge in graph.edges():
            v1, v2 = edge
            edge_weight = graph[v1][v2].get('weight', 1)
            A_data.append(edge_weight)
            A_row_index.append(node2idx[v1])
            A_col_index.append(node2idx[v2])

        # TODO: ???
        # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.sparse.csc_matrix.html
        A = sp.csc_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size))
        A_ = sp.csc_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                           shape=(node_size, node_size))

        D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
        L = D - A_
        return A, L


if __name__ == "__main__":
    # G = nx.read_edgelist('../../data/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    G = nx.karate_club_graph().to_directed()
    model = SDNE(G, hidden_size=[256, 128])
    model.train(batch_size=256, epochs=1, verbose=2)
    embeddings = model.get_embeddings()

    classify_embeddings_evaluate(embeddings, label_file="../../data/Wiki_labels.txt")
    # plot_embeddings(G, embeddings, path_file="../data/Wiki_labels.txt")
