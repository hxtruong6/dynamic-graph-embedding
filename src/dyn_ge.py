import json
from os.path import join, exists
import os
import tensorflow as tf

import networkx as nx
from math import ceil

import numpy as np
import pandas as pd

from src.static_ge import StaticGE
from src.utils.autoencoder import Autoencoder
from src.utils.visualize import plot_embedding


def get_hidden_layer(prop_size, input_dim, embedding_dim):
    hidden_dims = [input_dim]
    while ceil(hidden_dims[-1] * prop_size) > embedding_dim:
        hidden_dims.append(ceil(hidden_dims[-1] * prop_size))

    del hidden_dims[0]
    return hidden_dims


def handle_expand_model(model: Autoencoder, input_dim, net2net_applied=False, prop_size=0.3):
    if input_dim == model.get_input_dim():
        return model

    # NOTE: suppose just for addition nodes to graph
    model.expand_first_layer(layer_dim=input_dim)

    if not net2net_applied:
        return model

    layers_size = model.get_layers_size()
    index = 0
    while index < len(layers_size) - 1:
        layer_1_dim, layer_2_dim = layers_size[index]
        suitable_dim = ceil(layer_1_dim * prop_size)
        if suitable_dim > layer_2_dim:
            # the prev layer before embedding layer
            if index == len(layers_size) - 2:
                model.deeper(pos_layer=index)
                # model.info()
            else:
                added_size = suitable_dim - layer_2_dim
                model.wider(added_size=added_size, pos_layer=index)
                index += 1
        else:
            index += 1
        layers_size = model.get_layers_size()
    return model


def save_custom_model(model: Autoencoder, model_folder_path):
    folder_path, name = model_folder_path['folder_path'], model_folder_path['name']
    if not exists(folder_path):
        os.makedirs(folder_path)

    # TODO: how to use save_weights
    # model.save_weights(join(folder_path, name))
    save_weights_model(weights=model.get_weights_model(), filepath=join(folder_path, name + '_weights.json'))

    config_layer = model.get_config_layer()
    with open(join(folder_path, name + '.json'), 'w') as fi:
        json.dump(config_layer, fi)


def load_custom_model(model_folder_path):
    folder_path, name = model_folder_path['folder_path'], model_folder_path['name']
    with open(join(folder_path, name + '.json')) as fo:
        config_layer = json.load(fo)

    model = Autoencoder(
        input_dim=config_layer['input_dim'],
        embedding_dim=config_layer['embedding_dim'],
        hidden_dims=config_layer['hidden_dims'],
        v1=config_layer['l1'],
        v2=config_layer['l2']
    )
    # TODO: check model load_weight
    # model.load_weights(join(folder_path, name))
    weights = load_weights_model(filepath=join(folder_path, name + '_weights.json'))
    model.set_weights_model(weights=weights)

    return model


def save_weights_model(weights, filepath):
    pd.DataFrame(weights).to_json(filepath, orient='split')


def load_weights_model(filepath):
    weights = pd.read_json(filepath, orient='split').to_numpy()
    for layer_index in range(len(weights[0])):
        weights[0][layer_index][0] = np.array(weights[0][layer_index][0], dtype=np.float32)
        weights[0][layer_index][1] = np.array(weights[0][layer_index][1], dtype=np.float32)
        weights[1][layer_index][0] = np.array(weights[1][layer_index][0], dtype=np.float32)
        weights[1][layer_index][1] = np.array(weights[1][layer_index][1], dtype=np.float32)

    return weights


def get_hidden_dims(layers_size):
    hidden_dims = []
    for i, (l1, l2) in enumerate(layers_size):
        if i == 0:
            continue
        hidden_dims.append(l1)
    return hidden_dims


class DynGE(object):
    def __init__(self, graphs, embedding_dim, v1=0.001, v2=0.001):
        super(DynGE, self).__init__()
        if not graphs:
            raise ValueError("Must be provide graphs data")

        self.graphs = graphs
        self.graph_len = len(graphs)
        self.embedding_dim = embedding_dim
        self.v1 = v1
        self.v2 = v2
        self.static_ges = []
        self.model_folder_paths = []

    def get_all_embeddings(self):
        return [ge.get_embedding() for ge in self.static_ges]

    def get_embedding(self, index):
        if index < 0 or index >= self.graph_len:
            raise ValueError("index is invalid!")
        return self.static_ges[index].get_embedding()

    def train(self, prop_size=0.4, batch_size=64, epochs=100, filepath="../models/generate/", skip_print=5,
              net2net_applied=False, learning_rate=0.003):
        init_hidden_dims = get_hidden_layer(prop_size=prop_size, input_dim=len(self.graphs[0].nodes()),
                                            embedding_dim=self.embedding_dim)
        model = Autoencoder(
            input_dim=len(self.graphs[0].nodes()),
            embedding_dim=self.embedding_dim,
            hidden_dims=init_hidden_dims,
            v1=self.v1,
            v2=self.v2
        )
        ge = StaticGE(G=self.graphs[0], model=model)
        ge.train(batch_size=batch_size, epochs=epochs, skip_print=skip_print, learning_rate=learning_rate)
        # model.info(show_config=True)
        self.static_ges.append(ge)
        self.model_folder_paths.append({
            "folder_path": join(filepath, "graph_0"),
            "name": "graph_0"
        })
        save_custom_model(model=model, model_folder_path=self.model_folder_paths[0])

        for i in range(1, len(self.graphs)):
            graph = nx.Graph(self.graphs[i])
            input_dim = len(graph.nodes())
            prev_model = self._create_prev_model(index=i - 1)
            curr_model = handle_expand_model(model=prev_model, input_dim=input_dim,
                                             prop_size=prop_size, net2net_applied=net2net_applied)
            ge = StaticGE(G=graph, model=curr_model)
            ge.train(batch_size=batch_size, epochs=epochs, skip_print=skip_print, learning_rate=learning_rate)

            self.static_ges.append(ge)
            self.model_folder_paths.append({
                "folder_path": join(filepath, f"graph_{i}"),
                "name": f"graph_{i}"
            })
            save_custom_model(model=ge.get_model(), model_folder_path=self.model_folder_paths[i])

    def _create_prev_model(self, index):
        model = load_custom_model(self.model_folder_paths[index])
        return model

    def load_models(self, folder_path):
        model_folders_paths = os.listdir(folder_path)
        self.model_folder_paths = []
        self.static_ges = []
        for i, folder in enumerate(model_folders_paths):
            model_folder_path = {
                'folder_path': join(folder_path, folder),
                'name': folder
            }
            model = load_custom_model(model_folder_path=model_folder_path)
            self.model_folder_paths.append(model_folder_path)

            ge = StaticGE(G=self.graphs[i], model=model)
            self.static_ges.append(ge)

    def train_from_checkpoint(self, folder_path, batch_size=64, epochs=10, skip_print=10, learning_rate=0.001,
                              filepath=None):
        self.load_models(folder_path=folder_path)
        if filepath is None:
            filepath = folder_path

        for i in range(self.graph_len):
            self.static_ges[i].train(batch_size=batch_size, epochs=epochs, skip_print=skip_print,
                                     learning_rate=learning_rate)

            model_folder_path = {
                "folder_path": join(filepath, f"graph_{i}"),
                "name": f"graph_{i}"
            }
            save_custom_model(model=self.static_ges[i].get_model(), model_folder_path=model_folder_path)


if __name__ == "__main__":
    g1 = nx.complete_graph(100)
    g2 = nx.complete_graph(150)
    g3 = nx.complete_graph(180)
    graphs = [g1, g1, g1]
    dy_ge = DynGE(graphs=graphs, embedding_dim=4)
    # dy_ge.train(prop_size=0.4, epochs=300, skip_print=30, net2net_applied=False, learning_rate=0.0005,
    #             filepath="../models/generate/")
    dy_ge.load_models(folder_path="../models/generate")
    # dy_ge.train_from_checkpoint(folder_path="../models/generate/", filepath="../models/checkpoints_1", epochs=200,
    #                             skip_print=20, learning_rate=0.00005)

    embeddings = dy_ge.get_all_embeddings()
    for e in embeddings:
        plot_embedding(embedding=e)


