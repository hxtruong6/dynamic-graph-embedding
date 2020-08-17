import os
import pickle
import warnings
from os.path import isfile, join
from time import time

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec
import gensim.models.keyedvectors as word2vec

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph
from src.utils.link_pred_precision_k import check_link_predictionK
from src.utils.model_training_utils import create_folder, dyngem_alg, link_pred_eva, node2vec_alg, sdne_alg
from src.utils.data_utils import save_processed_data, load_single_processed_data
from src.utils.graph_util import print_graph_stats
from src.utils.link_prediction import preprocessing_graph_for_link_prediction
from src.utils.setting_param import SettingParam

warnings.filterwarnings("ignore")


# class DotDict(dict):
#     """
#     a dictionary that supports dot notation
#     as well as dictionary access notation
#     usage: d = DotDict() or d = DotDict({'val1':'first'})
#     set attributes: d.val2 = 'second' or d['val2'] = 'second'
#     get attributes: d.val2 or d['val2']
#     """
#     __getattr__ = dict.__getitem__
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
#
#     def __init__(self, dct):
#         super().__init__()
#         for key, value in dct.items():
#             if hasattr(value, 'keys'):
#                 value = DotDict(value)
#             self[key] = value


def load_embedding(filepath):
    with open(filepath, 'rb') as fp:
        embedding = pickle.load(fp)
    return embedding


def load_dy_embeddings(folder_path, index=None):
    print("Folder_path: ", folder_path)
    if not os.path.exists(folder_path):
        raise ValueError("Folder is invalid.")

    embeddings = []
    files = [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]
    length = len(files)
    for idx in range(length):
        if index is not None and idx != index:
            continue
        print("emb: ", join(folder_path, f"_{idx}"))
        embedding = load_embedding(filepath=join(folder_path, f"_{idx}"))
        embeddings.append(embedding)
    return embeddings


def load_node2vec_embeddings(graphs, folder_path, index=None):
    dy_embeddings = []
    for idx, graph in enumerate(graphs):
        if index is not None and idx != index:
            continue
        # node2vec = Node2Vec(graph=graph,
        #                     dimensions=1,
        #                     walk_length=1,
        #                     num_walks=1,
        #                     workers=2)  # Use temp_folder for big graphs
        # node2vec_model = node2vec.fit()
        # embedding = node2vec_model.wv.load_word2vec_format(join(folder_path, f"n2v_emb_{idx}"))
        # word2vec = Word2Vec.load_word2vec_format(join(folder_path, f"n2v_emb_{idx}"))
        embed_map = word2vec.KeyedVectors.load_word2vec_format(join(folder_path, f"n2v_emb_{idx}"), binary=False)
        embedding = [np.array(embed_map[str(u)]) for u in sorted(graph.nodes)]

        dy_embeddings.append(np.array(embedding))
    return dy_embeddings


if __name__ == "__main__":
    dataset_name = "soc_wiki"
    params = {
        # 'algorithm': {
        'is_dyge': False,
        'is_node2vec': True,
        'is_sdne': False,

        # 'folder_paths': {
        'dataset_folder': f"./data/{dataset_name}",
        'processed_link_pred_data_folder': f"./saved_data/processed_data/{dataset_name}_link_pred",

        'dyge_emb_folder': f"./saved_data/embeddings/{dataset_name}_link_pred",
        'node2vec_emb_folder': f"./saved_data/node2vec_emb/{dataset_name}_link_pred",
        'sdne_emb_folder': f"./saved_data/sdne_emb/{dataset_name}_link_pred",
    }

    params = SettingParam(**params)
    # ==================== Data =========================
    graphs, idx2node = read_dynamic_graph(
        folder_path=params.dataset_folder,
        limit=None,
        convert_to_idx=True
    )

    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)
    # =================== Processing data for link prediction ==========================
    print("Load processed data from disk...")
    g_hidden_df, g_hidden_partial = load_single_processed_data(folder=params.processed_link_pred_data_folder)

    # Set last graph in dynamic graph is hidden graph
    orginial_graph = graphs[-1]
    graphs[-1] = g_hidden_partial

    print("After processing for link prediction graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    # ========= DynGEM ===========
    if params.is_dyge:
        print("=============== DynGEM ============")
        # -------- Training ----------
        dy_embeddings = load_dy_embeddings(params.dyge_emb_folder, index=len(graphs) - 1)
        # link_pred_eva(g_hidden_df=g_hidden_df, hidden_dy_embedding=dy_embeddings[-1])
        k_query_res, AP = check_link_predictionK(embedding=dy_embeddings[-1], train_graph=g_hidden_partial,
                                                 origin_graph=orginial_graph,
                                                 k_query=[
                                                     2, 10,
                                                     100, 200, 1000, 10000
                                                 ])
        print(f"mAP = {AP}")
    # ============== Node2Vec ============
    if params.is_node2vec:
        print("=============== Node2vec ============")
        # Just need train last graph
        dy_embeddings = load_node2vec_embeddings(graphs, folder_path=params.node2vec_emb_folder, index=len(graphs) - 1)
        link_pred_eva(g_hidden_df=g_hidden_df, hidden_dy_embedding=dy_embeddings[-1])
        k_query_res, AP = check_link_predictionK(embedding=dy_embeddings[-1], train_graph=g_hidden_partial,
                                                 origin_graph=orginial_graph,
                                                 k_query=[
                                                     2, 10,
                                                     100, 200, 1000, 10000
                                                 ])
        print(f"mAP = {AP}")

    # == == == == == == == = SDNE == == == == == ==
    # if params.is_sdne:
    #     print("=============== SDNE ============")
    #     dy_embeddings = sdne_alg(
    #         graphs=graphs,
    #         params=params,
    #         index=len(graphs) - 1
    #     )
    #
    #     link_pred_eva(g_hidden_df=g_hidden_df, hidden_dy_embedding=dy_embeddings[0])
