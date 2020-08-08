import os
from os import listdir
from os.path import join, exists, isfile
from time import time
import numpy as np
import networkx as nx
import warnings
from node2vec import Node2Vec

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph, get_graph_from_file
from src.utils.checkpoint_config import CheckpointConfig
from src.utils.data_utils import load_processed_data, save_processed_data
from src.utils.stable_evaluate import stability_constant
from src.dyn_ge import TDynGE
from src.utils.graph_util import graph_to_graph_idx, print_graph_stats, draw_graph
from src.utils.link_prediction import preprocessing_graph_for_link_prediction, run_link_pred_evaluate, run_predict, \
    top_k_prediction_edges

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # ------------ Params -----------
    processed_data_folder = "../processed_data/cit_hepth"

    seed = 6
    # link prediction params
    show_acc_on_edge = True
    top_k = 10
    drop_node_percent = 0.35

    # ====================== Settings =================
    np.random.seed(seed=seed)
    if not exists(processed_data_folder):
        raise ValueError("Lack of processed data and trained model")
    # ==================== Data =========================
    graphs, idx2node = read_dynamic_graph(
        folder_path=folder_data,
        limit=None,
        convert_to_idx=True
    )
    # g1 = nx.gnm_random_graph(n=10, m=15, seed=6)
    # g2 = nx.gnm_random_graph(n=15, m=30, seed=6)
    # g3 = nx.gnm_random_graph(n=30, m=100, seed=6)
    #
    # graphs = [g1, g2, g3]

    # =============================================

    print("Number graphs: ", len(graphs))
    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)

    print("Load processed data from disk...")
    G_dfs, G_partial_list = load_processed_data(folder=processed_data_folder)
    # G_partial_list = graphs

    print("After processing for link prediction graphs:")
    for i, g in enumerate(G_partial_list):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    # ----------- Training area -----------
    dy_embeddings = []
    for g in G_partial_list:
        node2vec = Node2Vec(g, dimensions=128, walk_length=100, num_walks=2000,
                            workers=4)  # Use temp_folder for big graphs
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embedding = [model[str(u)] for u in sorted(g.nodes)]
        dy_embeddings.append(embedding)

    # -------- Stability constant -------------
    print(f"Stability constant= {stability_constant(graphs=G_partial_list, embeddings=dy_embeddings)}")

    # ----- run evaluate link prediction -------
    for i in range(len(graphs)):
        G_df = G_dfs[i]
        print(f"\n-->[Graph {i}] Run link predict evaluation ---")
        link_pred_model = run_link_pred_evaluate(
            graph_df=G_df,
            embeddings=dy_embeddings[i],
            num_boost_round=20000
        )
        possible_edges_df = G_df[G_df['link'] == 0]
        y_pred = run_predict(data=possible_edges_df, embedding=dy_embeddings[i], model=link_pred_model)
        top_k_edges = top_k_prediction_edges(
            G=graphs[i], y_pred=y_pred, possible_edges_df=possible_edges_df,
            top_k=top_k, show_acc_on_edge=show_acc_on_edge,
            plot_link_pred=False, limit_node=25,
        )
