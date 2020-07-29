import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model, regularizers, initializers
import numpy as np
import json

from src.utils.net2net import net2wider, net2deeper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class TPartCoder(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dims=None, l1=0.01, l2=0.01, seed=6):
        super(TPartCoder, self).__init__()
        self.layers = nn.ModuleList()
        # TODO: add regularization
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                layer = [
                    nn.Linear(in_features=input_dim, out_features=hidden_dims[i]),
                    nn.ReLU(),
                ]
            elif i == len(hidden_dims):
                layer = [
                    nn.Linear(in_features=hidden_dims[i - 1], out_features=output_dim),
                    nn.Sigmoid(),
                ]
            else:
                layer = [
                    nn.Linear(in_features=hidden_dims[i - 1], out_features=hidden_dims[i]),
                    nn.ReLU(),
                ]
            self.layers.extend(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_hidden_dims(self):
        hidden_dims = []
        for layer in self.layers:
            if type(layer) == nn.Linear:
                hidden_dims.append(layer.in_features)
        return hidden_dims[1:]

    def info(self, show_weights=False):
        print(f"Number of layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                print(f"Layer {i + 1} \t Shape =({layer.in_features},{layer.out_features})")
                if show_weights:
                    print(f"\tWeight= {layer.weight.data} \n\tBias= {layer.bias.data}")

    def insert_first_layer(self, layer_dim):
        out_feature = self.layers[0].in_features
        layer = nn.Linear(in_features=layer_dim, out_features=out_feature).apply(weights_init)
        self.layers.insert(0, nn.ReLU())
        self.layers.insert(0, layer)

    def insert_last_layer(self, layer_dim):
        in_feature = self.layers[-2].out_features  # self.layers[-1] is Sigmoid activation
        layer = [
            nn.Linear(in_features=in_feature, out_features=layer_dim).apply(weights_init),
            nn.Sigmoid()
        ]
        self.layers[-1] = nn.ReLU()
        self.layers.extend(layer)


class TAutoencoder(nn.Module):
    def __init__(self, input_dim=None, embedding_dim=None, hidden_dims=None, l1=0.01, l2=0.01):
        super(TAutoencoder, self).__init__()
        if input_dim is None or embedding_dim is None:
            return
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.l1 = l1
        self.l2 = l2

        # ======== Create layers
        self.encoder = TPartCoder(input_dim=input_dim, output_dim=embedding_dim, hidden_dims=hidden_dims, l1=l1, l2=l2)
        self.encoder.apply(weights_init)

        self.decoder = TPartCoder(input_dim=embedding_dim, output_dim=input_dim, hidden_dims=hidden_dims[::-1], l1=l1,
                                  l2=l2)
        self.decoder.apply(weights_init)

    def forward(self, x):
        y = self.encoder(x)
        return self.decoder(y), y

    def get_embedding(self, x):
        embedding = self.encoder(x)
        embedding = embedding.clone().detach().cpu()
        return embedding.detach().numpy()

    def get_reconstruction(self, x):
        reconstruction = self.decoder(self.encoder(x))
        reconstruction = reconstruction.clone().detach().cpu()
        return reconstruction.detach().numpy()

    def get_hidden_dims(self):
        '''
        Suppose encoder part and decoder part have symmetric size. So just return encoder part.
        Decoder part is reverse of encoder part
        :param half_model:
        :return:
        '''
        hidden_dims = self.encoder.get_hidden_dims()
        return hidden_dims

    def info(self, show_weights=False):
        self.encoder.info(show_weights=show_weights)
        self.decoder.info(show_weights=show_weights)

    def expand_first_layer(self, layer_dim):
        self.input_dim = layer_dim
        self.encoder.insert_first_layer(layer_dim=layer_dim)
        self.decoder.insert_last_layer(layer_dim=layer_dim)
        self.hidden_dims = self.get_hidden_dims()

    def get_input_dim(self):
        return self.input_dim

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_config_layer(self):
        config_layer = {
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "l1": self.l1,
            "l2": self.l2
        }
        return config_layer


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6)

    num_epochs = 1
    dataset = torch.randn(5, 3)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    #  create dataset

    ae = TAutoencoder(input_dim=3, embedding_dim=2, hidden_dims=[4]).to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for data in dataloader:
            inp = data
            print(inp)
            inp = Variable(inp).to(device)
            # ===================forward=====================
            output, embed_out = ae(inp)
            loss = criterion(output, inp)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))

    print(ae.get_hidden_dims())
    ae.info(show_weights=True)

    print("\nExpand autoencoder")
    ae.expand_first_layer(layer_dim=6)
    ae.info(show_weights=True)
    print(ae)
