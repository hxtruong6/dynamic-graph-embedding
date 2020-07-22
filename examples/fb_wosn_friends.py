import os
from os import listdir
from os.path import join, exists, isfile
from time import time
import numpy as np
import networkx as nx
import pandas as pd
import warnings

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph, get_graph_from_file

warnings.filterwarnings("ignore")

from src.dyn_ge import DynGE
from src.utils.graph_util import graph_to_graph_idx, print_graph_stats, draw_graph
from src.utils.link_prediction import preprocessing_graph_for_link_prediction, run_link_pred_evaluate, run_predict, \
    top_k_prediction_edges


def save_processed_graph(graph, folder, index):
    if not exists(folder):
        os.makedirs(folder)
    nx.write_edgelist(graph, f'{folder}/graph{index}.edgelist.', data=False)


def save_graph_df(graph_df: pd.DataFrame, folder, index):
    if not exists(folder):
        os.makedirs(folder)
    graph_df.to_json(join(folder, f"graph_{index}.json.gz"))
    # graph_df.to_csv(join(folder, f"graph_{index}.csv"), index=False)


def load_graphs_df(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    graphs_df = []
    for f in sorted(files):
        graphs_df.append(
            # pd.read_csv(join(folder, f))
            pd.read_json(join(folder, f), compression='gzip')
        )
    return graphs_df


def load_processed_graphs(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    graphs = []
    for f in sorted(files):
        graphs.append(
            get_graph_from_file(join(folder, f))
        )
    return graphs


# https://stackoverflow.com/questions/39450065/python-3-read-write-compressed-json-objects-from-to-gzip-file

if __name__ == "__main__":
    # ------------ Params -----------
    folder_data = "../data/fb"
    processed_data_folder = "../processed_data/fb"
    weight_model_folder = "../models/fb"
    load_model = False
    load_processed_data = False
    epochs = 1
    skip_print = 5
    batch_size = 256
    seed = 6

    # link prediction params
    show_acc_on_edge = True
    top_k = 10
    drop_node_percent = 0.5

    # ====================== Settings =================
    np.random.seed(seed=seed)
    if not exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    if not exists(weight_model_folder):
        os.makedirs(weight_model_folder)

    # =============================================
    graphs, idx2node = read_dynamic_graph(folder_path=folder_data, limit=2, convert_to_idx=True)
    # g1 = nx.gnm_random_graph(n=30, m=60, seed=6)
    # g2 = nx.gnm_random_graph(n=30, m=70, seed=6)
    # original_graphs = [g1]

    print("Number graphs: ", len(graphs))
    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i)
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)

    if load_processed_data:
        print("Load processed data from disk...")
        G_dfs = load_graphs_df(folder=join(processed_data_folder, "dfs"))
        G_partial_list = load_processed_graphs(folder=join(processed_data_folder, "graphs"))
    else:
        print("\n[ALL] Pre-processing graph for link prediction...")
        start_time = time()
        G_dfs = []
        G_partial_list = []
        for idx, g in enumerate(graphs):
            print(f"==== Graph {idx}: ")
            g_df, g_partial = preprocessing_graph_for_link_prediction(G=g, k_length=2,
                                                                      drop_node_percent=drop_node_percent)
            G_dfs.append(g_df)
            G_partial_list.append(g_partial)
            # Save processed data. NOTE: save idx graph. Not original graph
            save_graph_df(g_df, folder=join(processed_data_folder, "dfs"), index=idx)
            save_processed_graph(g_partial, folder=join(processed_data_folder, "graphs"), index=idx)
            # draw_graph(g=g_partial, limit_node=25)

        print(f"[ALL] Processed in {round(time() - start_time, 2)}s\n")

    print("After processing for link prediction graphs:")
    for i, g in enumerate(G_partial_list):
        print_graph_stats(g, i)

    # -------------------------------
    dy_ge = DynGE(graphs=G_partial_list, embedding_dim=4)
    if load_model and weight_model_folder:
        print("\n-----------\nStart load model...")
        start_time = time()
        dy_ge.load_models(folder_path=weight_model_folder)
        print(f"Loaded model: {round(time() - start_time, 2)}s\n--------------\n")

    else:
        print("\n-----------\nStart total training...")
        start_time = time()
        dy_ge.train(prop_size=0.3, epochs=epochs, skip_print=skip_print, net2net_applied=False,
                    learning_rate=0.0001,
                    batch_size=batch_size, filepath=weight_model_folder, save_model_point=None)
        print(f"Finish total training: {round(time() - start_time, 2)}s\n--------------\n")

    dy_embeddings = dy_ge.get_all_embeddings()

    # ----- run evaluate link prediction -------
    for i in range(len(graphs)):
        G_df = G_dfs[i]
        link_pred_model = run_link_pred_evaluate(graph_df=G_df, embedding=dy_embeddings[i], num_boost_round=20000)
        possible_edges_df = G_df[G_df['link'] == 0]
        y_pred = run_predict(data=possible_edges_df, embedding=dy_embeddings[i], model=link_pred_model)
        top_k_edges = top_k_prediction_edges(G=graphs[i], y_pred=y_pred, possible_edges_df=possible_edges_df,
                                             top_k=top_k, show_acc_on_edge=show_acc_on_edge, plot_link_pred=True,
                                             limit_node=25)
