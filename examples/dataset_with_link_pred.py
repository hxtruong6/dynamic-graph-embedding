import warnings
from time import time

import networkx as nx
import numpy as np

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph
from src.utils.model_training_utils import create_folder, dyngem_alg, link_pred_eva, node2vec_alg
from src.utils.data_utils import save_processed_data, load_single_processed_data
from src.utils.graph_util import print_graph_stats
from src.utils.link_prediction import preprocessing_graph_for_link_prediction
from src.utils.setting_param import SettingParam

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    dataset_name = "soc_wiki"
    params = {
        # 'algorithm': {
        'is_dyge': True,
        'is_node2vec': False,

        # 'folder_paths': {
        'dataset_folder': f"./data/{dataset_name}",
        'processed_link_pred_data_folder': f"./saved_data/processed_data/{dataset_name}_link_pred",
        'dyge_weight_folder': f"./saved_data/models/{dataset_name}_link_pred",
        'dyge_emb_folder': f"./saved_data/embeddings/{dataset_name}_link_pred",
        'node2vec_emb_folder': f"./saved_data/node2vec_emb/{dataset_name}_link_pred",
        'global_seed': 6,

        # 'training_config': {
        'is_load_link_pred_data': False,
        'is_load_dyge_model': False,
        'specific_dyge_model_index': None,
        'dyge_resume_training': False,

        'is_load_n2v_model': False,

        # 'dyge_config': {
        'prop_size': 0.3,
        'embedding_dim': 100,
        'epochs': 2,
        'skip_print': 200,
        'batch_size': 512,  # 512
        'early_stop': 100,  # 100
        'learning_rate_list': [
            0.0005,
            # 3e-4, 1e-4,
            # 5e-5, 3e-5, 1e-5,
            # 5e-6, 1e-6
        ],
        'alpha': 0.2,
        'beta': 10,
        'l1': 0.001,
        'l2': 0.0005,
        'net2net_applied': False,
        'ck_length_saving': 50,
        'ck_folder': f'./saved_data/models/{dataset_name}_link_pred_ck',
        'dyge_shuffle': True,

        # 'link_pred_config': {
        'show_acc_on_edge': True,
        'top_k': 10,
        'drop_node_percent': 0.2,
    }

    params = SettingParam(**params)

    # ====================== Settings =================
    np.random.seed(seed=params.global_seed)

    create_folder(params.processed_link_pred_data_folder)
    create_folder(params.dyge_weight_folder)
    create_folder(params.dyge_emb_folder)
    create_folder(params.node2vec_emb_folder)
    # ==================== Data =========================

    # graphs, idx2node = read_dynamic_graph(
    #     folder_path=params.dataset_folder,
    #     limit=None,
    #     convert_to_idx=True
    # )
    g1 = nx.gnm_random_graph(n=10, m=15, seed=6)
    g2 = nx.gnm_random_graph(n=15, m=30, seed=6)
    g3 = nx.gnm_random_graph(n=30, m=100, seed=6)
    graphs = [g1, g2, g3]

    print("Number graphs: ", len(graphs))
    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)
    # =================== Processing data for link prediction ==========================

    if params.is_load_link_pred_data:
        print("Load processed data from disk...")
        g_hidden_df, g_hidden_partial = load_single_processed_data(folder=params.processed_link_pred_data_folder)
    else:
        print("\n[ALL] Pre-processing graph for link prediction...")
        start_time = time()
        print(f"==== Graph {len(graphs)}: ")
        g_hidden_df, g_hidden_partial = preprocessing_graph_for_link_prediction(
            G=graphs[-1],
            drop_node_percent=params.drop_node_percent,
            edge_rate=0.1
        )
        # NOTE: save idx graph. Not original graph
        save_processed_data(g_hidden_df, g_hidden_partial, folder=params.processed_link_pred_data_folder,
                            index=len(graphs))
        # draw_graph(g=g_partial, limit_node=25)
        print(f"[ALL] Processed in {round(time() - start_time, 2)}s\n")

    # Set last graph in dynamic graph is hidden graph
    graphs[-1] = g_hidden_partial
    print("After processing for link prediction graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    # ========= DynGEM ===========

    if params.is_dyge:
        print("=============== DynGEM ============")
        # -------- Training ----------
        dy_ge, dy_embeddings = dyngem_alg(graphs=graphs, params=params)
        # Just use link prediction for last hidden graph
        hidden_dy_embedding = dy_embeddings[-1]
        link_pred_eva(g_hidden_df=g_hidden_df, hidden_dy_embedding=hidden_dy_embedding)
    # ============== Node2Vec ============
    if params.is_node2vec:
        print("=============== Node2vec ============")
        # Just need train last graph
        hidden_embedding = node2vec_alg(
            graphs=graphs,
            embedding_dim=params.embedding_dim,
            index=len(graphs) - 1,
            folder_path=params.node2vec_emb_folder
        )
        link_pred_eva(g_hidden_df=g_hidden_df, hidden_dy_embedding=hidden_embedding)
