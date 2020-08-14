import json
from os.path import join, exists
import os
from math import ceil
import numpy as np
import pandas as pd
import torch

from src.utils.autoencoder import TAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hidden_layer(prop_size, input_dim, embedding_dim):
    hidden_dims = [input_dim]
    while ceil(hidden_dims[-1] * prop_size) > embedding_dim:
        hidden_dims.append(ceil(hidden_dims[-1] * prop_size))

    del hidden_dims[0]
    if len(hidden_dims) == 0:
        hidden_dims.append(int((input_dim - embedding_dim) / 2 + embedding_dim))
    return hidden_dims


def handle_expand_model(model: TAutoencoder, input_dim, net2net_applied=False, prop_size=0.3):
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


def save_custom_model(model: TAutoencoder, filepath, checkpoint=None, compress=True):
    # if checkpoint is not None:
    #     if folder_path[-1] == '/':
    #         folder_path = folder_path[:-1]
    #
    #     r_pos = folder_path.rfind('/')
    #     begin_folder_path = folder_path[:r_pos] + "__ck_" + str(checkpoint)
    #     if not exists(begin_folder_path):
    #         os.makedirs(begin_folder_path)
    #
    #     folder_path = begin_folder_path + folder_path[r_pos:]
    #
    # if not exists(folder_fpath):
    #     os.makedirs(folder_path)

    folder_path = filepath[:filepath.rfind('/')]
    if not exists(folder_path):
        os.makedirs(folder_path)

    model_path = filepath + ".pt"
    torch.save(model.state_dict(), model_path)

    config_layer = model.get_config_layer()
    config_path = filepath + ".json"
    with open(config_path, 'w') as fi:
        json.dump(config_layer, fi, indent=2)


def load_custom_model(filepath):
    # folder_path, name = model_folder_path['folder_path'], model_folder_path['name']
    config_path = filepath + ".json"
    with open(config_path) as fo:
        config_layer = json.load(fo)

    model = TAutoencoder(
        input_dim=config_layer['input_dim'],
        embedding_dim=config_layer['embedding_dim'],
        hidden_dims=config_layer['hidden_dims'],
        l1=config_layer['l1'],
        l2=config_layer['l2'],
        activation=config_layer['activation']
    )

    model_path = filepath + ".pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def save_weights_model(weights, filepath, compress=True):
    if filepath[-3:] != ".gz":
        filepath += ".gz"
    pd.DataFrame(weights).to_json(filepath, compression='gzip')


def load_weights_model(filepath):
    if filepath[-3:] != ".gz":
        filepath += ".gz"
    weights = pd.read_json(filepath, compression='gzip').to_numpy()
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
