import os
from os import listdir
from os.path import join, exists, isfile
from time import time
import numpy as np
import networkx as nx
import pandas as pd
import warnings

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph, get_graph_from_file
from src.utils.checkpoint_config import CheckpointConfig

warnings.filterwarnings("ignore")

from src.dyn_ge import TDynGE
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


if __name__ == "__main__":
    def train_model():
        print("\n-----------\nStart total training...")
        start_time_train = time()
        dy_ge.train(prop_size=prop_size, epochs=epochs, skip_print=skip_print,
                    net2net_applied=net2net_applied, learning_rate=learning_rate,
                    batch_size=batch_size, folder_path=weight_model_folder,
                    checkpoint_config=checkpoint_config, from_loaded_model=train_from_loaded_model,
                    early_stop=early_stop)
        print(f"Finish total training: {round(time() - start_time_train, 2)}s\n--------------\n")


    # ------------ Params -----------
    folder_data = "../data/cit_hepth"
    processed_data_folder = "../processed_data/cit_hepth"
    weight_model_folder = "../models/cit_hepth"
    load_model = False
    load_processed_data = True
    train_from_loaded_model = False
    epochs = 20
    skip_print = 1
    batch_size = 256
    early_stop = 5 # 100
    seed = 6
    prop_size = 0.35
    learning_rate = 0.0005
    alpha = 0.01
    beta = 2
    l1 = 0.001
    l2 = 0.0005
    embedding_dim = 128
    net2net_applied = False
    checkpoint_config = CheckpointConfig(number_saved=50, folder_path=weight_model_folder + "_ck")

    # link prediction params
    show_acc_on_edge = True
    top_k = 10
    drop_node_percent = 0.3

    # ====================== Settings =================
    np.random.seed(seed=seed)
    if not exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    if not exists(weight_model_folder):
        os.makedirs(weight_model_folder)

    # =============================================
    graphs, idx2node = read_dynamic_graph(
        folder_path=folder_data,
        limit=1,
        convert_to_idx=True
    )
    # g1 = nx.gnm_random_graph(n=30, m=100, seed=6)
    # g2 = nx.gnm_random_graph(n=60, m=200, seed=6)
    # graphs = [g1, g2]

    print("Number graphs: ", len(graphs))
    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)
    # TODO:[BUG] loaded graph has smaller number of node then original
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
            g_df, g_partial = preprocessing_graph_for_link_prediction(
                G=g, k_length=2,
                drop_node_percent=drop_node_percent
            )
            G_dfs.append(g_df)
            G_partial_list.append(g_partial)
            # Save processed data. NOTE: save idx graph. Not original graph
            save_graph_df(g_df, folder=join(processed_data_folder, "dfs"), index=idx)
            save_processed_graph(g_partial, folder=join(processed_data_folder, "graphs"), index=idx)
            # draw_graph(g=g_partial, limit_node=25)

        print(f"[ALL] Processed in {round(time() - start_time, 2)}s\n")

    print("After processing for link prediction graphs:")
    for i, g in enumerate(G_partial_list):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    # -------------------------------
    dy_ge = TDynGE(
        graphs=G_partial_list, embedding_dim=embedding_dim,
        alpha=alpha, beta=beta, l1=l1, l2=l2
    )
    if load_model and weight_model_folder:
        if train_from_loaded_model:
            train_model()
        else:
            print("\n-----------\nStart load model...")
            start_time = time()
            dy_ge.load_models(folder_path=weight_model_folder)
            print(f"Loaded model: {round(time() - start_time, 2)}s\n--------------\n")
    else:
        # if load_model==False => train_from_loaded_model = False
        train_from_loaded_model = False
        train_model()

    dy_embeddings = dy_ge.get_all_embeddings()

    # ----- run evaluate link prediction -------
    for i in range(len(graphs)):
        G_df = G_dfs[i]
        link_pred_model = run_link_pred_evaluate(
            graph_df=G_df,
            embeddings=dy_embeddings[i],
            num_boost_round=20000
        )
        possible_edges_df = G_df[G_df['link'] == 0]
        y_pred = run_predict(data=possible_edges_df, embedding=dy_embeddings[i], model=link_pred_model)
        top_k_edges = top_k_prediction_edges(
            G=graphs[i], y_pred=y_pred, possible_edges_df=possible_edges_df,
            top_k=top_k, show_acc_on_edge=show_acc_on_edge, plot_link_pred=True,
            limit_node=25
        )
    # print(dy_embeddings[0][:10,:10])
