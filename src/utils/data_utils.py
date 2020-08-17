import os
import pickle
from os import listdir, makedirs
from os.path import isfile, join, exists
import pandas as pd
import networkx as nx
import numpy as np
import gensim.models.keyedvectors as word2vec

from src.data_preprocessing.graph_preprocessing import get_graph_from_file


def load_processed_data(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    graphs_df = []
    graphs = []
    for f in sorted(files):
        if ".json" in f:
            graphs_df.append(pd.read_json(join(folder, f)))
        else:  # ".edgelist" in f is True
            graphs.append(get_graph_from_file(join(folder, f)))
    return graphs_df, graphs


def load_single_processed_data(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    graph_df = None
    graph = None
    for f in sorted(files):
        if ".json" in f:
            graph_df = pd.read_json(join(folder, f))
        else:  # ".edgelist" in f is True
            # graph = get_graph_from_file(join(folder, f))
            graph = nx.read_gpickle(join(folder, f))
    return graph_df, graph


def save_processed_data(graph_df: pd.DataFrame, graph: nx.Graph, folder, index):
    if not exists(folder):
        makedirs(folder)
    # nx.write_edgelist(graph, f'{folder}/graph{index}.edgelist', data=False)
    nx.write_gpickle(graph, f'{folder}/graph_{index}.gpickle')
    graph_df.to_json(join(folder, f"graph_{index}.json"))


def load_embedding(filepath):
    with open(filepath, 'rb') as fp:
        embedding = pickle.load(fp)
    return embedding


def load_dy_embeddings(folder_path, index=None):
    print("Folder_path: ", folder_path)
    if not os.path.exists(folder_path):
        raise ValueError("Folder is invalid.")

    if index is not None:
        file = join(folder_path, f"_{index}")
        return [load_embedding(filepath=file)]

    embeddings = []
    files = [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]
    length = len(files)
    for idx in range(length):
        print("emb: ", join(folder_path, f"_{idx}"))
        embedding = load_embedding(filepath=join(folder_path, f"_{idx}"))
        embeddings.append(embedding)
    return embeddings


def load_node2vec_embeddings(graphs, folder_path, index=None):
    dy_embeddings = []
    for idx, graph in enumerate(graphs):
        if index is not None and idx != index:
            continue
        embed_map = word2vec.KeyedVectors.load_word2vec_format(join(folder_path, f"n2v_emb_{idx}"), binary=True)
        embedding = [np.array(embed_map[str(u)]) for u in sorted(graph.nodes)]

        dy_embeddings.append(np.array(embedding))
    return dy_embeddings
