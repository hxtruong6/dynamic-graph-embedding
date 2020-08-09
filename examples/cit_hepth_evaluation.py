import os
from os import listdir
from os.path import join, exists, isfile
from time import time
import numpy as np
import networkx as nx
import warnings

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
    def check_current_loss_model():
        for model_idx in range(len(G_partial_list)):
            dy_ge.train_at(model_index=model_idx, folder_path=weight_model_folder,
                           epochs=1, learning_rate=1e-7,
                           is_load_from_previous_model=False)


    # ----------- Run part --------------
    is_run_stability_constant = False
    is_run_evaluate_link_prediction = False
    # ------------ Params -----------
    folder_data = "./data/cit_hepth"
    processed_data_folder = "./processed_data/cit_hepth"
    weight_model_folder = "./models/cit_hepth"
    embeddings_folder = "./embeddings/cit_hepth"

    seed = 6
    alpha = 0.2
    beta = 10
    l1 = 0.001
    l2 = 0.0005
    embedding_dim = 128
    # link prediction params
    show_acc_on_edge = True
    top_k = 10
    drop_node_percent = 0.35

    # ====================== Settings =================
    np.random.seed(seed=seed)
    if not exists(processed_data_folder) or not exists(weight_model_folder):
        raise ValueError("Lack of processed data and trained model")
    # ==================== Data =========================
    graphs, idx2node = read_dynamic_graph(
        folder_path=folder_data,
        limit=None,
        convert_to_idx=True
    )
    # =============================================

    print("Number graphs: ", len(graphs))
    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)

    print("Load processed data from disk...")
    G_dfs, G_partial_list = load_processed_data(folder=processed_data_folder)

    print("After processing for link prediction graphs:")
    for i, g in enumerate(G_partial_list):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    # ----------- Training area -----------
    dy_ge = TDynGE(
        graphs=G_partial_list, embedding_dim=embedding_dim,
        alpha=alpha, beta=beta, l1=l1, l2=l2
    )

    print("\n-----------\nStart load model...")
    dy_ge.load_models(folder_path=weight_model_folder)

    # Uncomment to know current loss value
    print("Check current loss value of model: ")
    check_current_loss_model()

    # print("Saving embedding... ")
    # dy_ge.save_embeddings(folder_path=embeddings_folder)
    print("Loading embedding...", end=" ")
    start_time = time()
    dy_embeddings = dy_ge.load_embeddings(folder_path=embeddings_folder)
    # dy_embeddings = dy_ge.get_all_embeddings()
    print(f"{round(time() - start_time, 2)}s")

    # -------- Stability constant -------------
    if is_run_stability_constant:
        print(f"Stability constant= {stability_constant(graphs=G_partial_list, embeddings=dy_embeddings)}")

    # ----- run evaluate link prediction -------
    if is_run_evaluate_link_prediction:
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
