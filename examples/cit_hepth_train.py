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
from src.utils.data_utils import load_processed_data, save_processed_data
from src.utils.stable_evaluate import stability_constant

warnings.filterwarnings("ignore")

from src.dyn_ge import TDynGE
from src.utils.graph_util import graph_to_graph_idx, print_graph_stats, draw_graph
from src.utils.link_prediction import preprocessing_graph_for_link_prediction, run_link_pred_evaluate, run_predict, \
    top_k_prediction_edges

if __name__ == "__main__":
    # If model_idx -> must be train from previous model
    def train_overall(lr):
        '''
        Method to create saved model on disk for later optimizer
        :return:
        '''
        dy_ge.train(prop_size=prop_size, epochs=1, skip_print=1,
                    net2net_applied=net2net_applied, learning_rate=lr,
                    batch_size=batch_size, folder_path=weight_model_folder)


    def train_model():
        print("\n-----------\nStart total training...")

        print("Overall training")
        train_overall(lr=learning_rate_list[0])

        print("\n### ==\tOptimize model training == ###")

        start_time_train = time()
        for model_idx in range(len(G_partial_list)):
            print(f"\n==========\t Model index = {model_idx} ============")
            for i, lr in enumerate(learning_rate_list):
                is_load_from_previous_model = False
                if i == 0:
                    is_load_from_previous_model = True  # Always create from previous model if start training
                print("\tLearning rate = ", lr)
                dy_ge.train_at(model_index=model_idx,
                               prop_size=prop_size, epochs=epochs, skip_print=skip_print,
                               net2net_applied=net2net_applied, learning_rate=lr,
                               batch_size=batch_size, folder_path=weight_model_folder,
                               ck_config=checkpoint_config, early_stop=early_stop,
                               is_load_from_previous_model=is_load_from_previous_model)

        print(f"\nFinish total training: {round(time() - start_time_train, 2)}s\n--------------\n")


    def train_model_at_index():
        '''
        TODO: Should support for resuming train which continue train with learning_rate
        :return:
        '''
        print(f"\n==========\t Model index = {specific_model_index} ============")
        start_time_train = time()
        for i, lr in enumerate(learning_rate_list):
            is_load_from_previous_model = False
            if i == 0:
                is_load_from_previous_model = True  # Prevent re-train model
            print("\tLearning rate = ", lr)
            dy_ge.train_at(model_index=specific_model_index,
                           prop_size=prop_size, epochs=epochs, skip_print=skip_print,
                           net2net_applied=net2net_applied, learning_rate=lr,
                           batch_size=batch_size, folder_path=weight_model_folder,
                           ck_config=checkpoint_config, early_stop=early_stop,
                           is_load_from_previous_model=is_load_from_previous_model)

        print(f"\nFinish total training: {round(time() - start_time_train, 2)}s\n--------------\n")


    def check_current_loss_model():
        for model_idx in range(len(G_partial_list)):
            dy_ge.train_at(model_index=model_idx,
                           prop_size=prop_size, epochs=1, skip_print=skip_print,
                           net2net_applied=net2net_applied, learning_rate=1e-6,
                           batch_size=batch_size, folder_path=weight_model_folder,
                           ck_config=checkpoint_config, early_stop=early_stop,
                           is_load_from_previous_model=False)


    # ----------- Run part --------------
    is_load_processed_data = True
    is_just_load_model = True  # Set True to run evaluate faster. If not: train->evaluate
    is_run_stability_constant = False
    is_run_evaluate_link_prediction = False
    # ------------ Params -----------
    folder_data = "../data/cit_hepth"
    processed_data_folder = "../processed_data/cit_hepth"
    weight_model_folder = "../models/cit_hepth"

    epochs = 200
    skip_print = 20
    batch_size = 512  # 512
    early_stop = 100  # 100
    seed = 6
    prop_size = 0.35

    # empty if not need to train with list lr else priority
    learning_rate_list = [
        0.0005, 1e-4, 1e-4,
        # 5e-5, 3e-5, 1e-5,
        # 5e-6, 1e-6
    ]
    # learning_rate_list = [0.01]  # empty if not need to train with list lr else priority

    # Default=None index here if you want to train just one model. If not it will run all dataset
    specific_model_index = None  # must be order: 0,1,2,..

    alpha = 0.2
    beta = 10
    l1 = 0.001
    l2 = 0.0005
    embedding_dim = 128
    net2net_applied = False
    checkpoint_config = CheckpointConfig(number_saved=50, folder_path=weight_model_folder + "_ck")

    # link prediction params
    show_acc_on_edge = True
    top_k = 10
    drop_node_percent = 0.35

    # ====================== Settings =================
    np.random.seed(seed=seed)
    if not exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    if not exists(weight_model_folder):
        os.makedirs(weight_model_folder)

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

    if is_load_processed_data:
        print("Load processed data from disk...")
        G_dfs, G_partial_list = load_processed_data(folder=processed_data_folder)
    else:
        print("\n[ALL] Pre-processing graph for link prediction...")
        G_dfs = []
        G_partial_list = []
        start_time = time()
        for idx, g in enumerate(graphs):
            print(f"==== Graph {idx}: ")
            g_df, g_partial = preprocessing_graph_for_link_prediction(
                G=g, k_length=2,
                drop_node_percent=drop_node_percent
            )
            G_dfs.append(g_df)
            G_partial_list.append(g_partial)

            # Save processed data.
            # NOTE: save idx graph. Not original graph
            save_processed_data(g_df, g_partial, folder=processed_data_folder, index=idx)
            # draw_graph(g=g_partial, limit_node=25)

        print(f"[ALL] Processed in {round(time() - start_time, 2)}s\n")

    print("After processing for link prediction graphs:")
    for i, g in enumerate(G_partial_list):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    # ----------- Training area -----------
    dy_ge = TDynGE(
        graphs=G_partial_list, embedding_dim=embedding_dim,
        alpha=alpha, beta=beta, l1=l1, l2=l2
    )
    if is_just_load_model:
        print("\n-----------\nStart load model...")
        start_time = time()
        dy_ge.load_models(folder_path=weight_model_folder)
        print(f"Loaded model: {round(time() - start_time, 2)}s\n--------------\n")
    elif specific_model_index is not None:
        train_model_at_index()
    else:
        train_model()

    # Uncomment to know current loss value
    check_current_loss_model()

    # -------- Stability constant -------------
    if is_run_stability_constant:
        dy_embeddings = dy_ge.get_all_embeddings()
        print(f"Stability constant= {stability_constant(graphs=G_partial_list, embeddings=dy_embeddings)}")

    # ----- run evaluate link prediction -------
    if is_run_evaluate_link_prediction:
        dy_embeddings = dy_ge.get_all_embeddings()
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
                plot_link_pred=True, limit_node=25
            )
