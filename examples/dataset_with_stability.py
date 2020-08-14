import warnings

import networkx as nx
import numpy as np

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph
from src.utils.model_training_utils import create_folder, dyngem_alg, node2vec_alg, sdne_alg
from src.utils.graph_util import print_graph_stats
from src.utils.setting_param import SettingParam
from src.utils.stable_evaluate import stability_constant

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    dataset_name = "cit_hepth"
    params = {
        # 'algorithm': {
        'is_dyge': False,
        'is_node2vec': False,
        'is_sdne': True,

        # 'folder_paths': {
        'dataset_folder': f"./data/{dataset_name}",
        'processed_link_pred_data_folder': f"./saved_data/processed_data/{dataset_name}_stability",
        'dyge_weight_folder': f"./saved_data/models/{dataset_name}_stability",
        'dyge_emb_folder': f"./saved_data/embeddings/{dataset_name}_stability",
        'node2vec_emb_folder': f"./saved_data/node2vec_emb/{dataset_name}_stability",
        'global_seed': 6,

        # 'training_config': {
        'is_load_dyge_model': False,
        'specific_dyge_model_index': None,
        'dyge_resume_training': False,

        'is_load_n2v_model': False,

        # 'dyge_config': {
        'prop_size': 0.3,
        'embedding_dim': 100,
        'epochs': 200,
        'skip_print': 20,
        'batch_size': None,  # 512
        'early_stop': 200,  # 100
        'learning_rate_list': [
            0.0005,
            3e-4, 1e-4,
            5e-5, 3e-5, 1e-5,
            5e-6, 1e-6
        ],
        'alpha': 0.2,
        'beta': 10,
        'l1': 0.001,
        'l2': 0.0005,
        'net2net_applied': False,
        'ck_length_saving': 50,
        'ck_folder': f'./saved_data/models/{dataset_name}_stability_ck',
        'dyge_shuffle': True,

        # SDNE
        'sdne_learning_rate': 5e-5,
        'sdne_shuffle': True,

        # 'link_pred_config': {
        'show_acc_on_edge': True,
        'top_k': 10,
        'drop_node_percent': 0.2,
    }

    params = SettingParam(**params)

    # ====================== Settings =================
    np.random.seed(seed=params.global_seed)

    # create_folder(params.processed_link_pred_data_folder)
    create_folder(params.dyge_weight_folder)
    create_folder(params.dyge_emb_folder)
    create_folder(params.node2vec_emb_folder)
    # ==================== Data =========================

    graphs, idx2node = read_dynamic_graph(
        folder_path=params.dataset_folder,
        limit=None,
        convert_to_idx=True
    )
    # g1 = nx.gnm_random_graph(n=10, m=15, seed=6)
    # g2 = nx.gnm_random_graph(n=15, m=30, seed=6)
    # g3 = nx.gnm_random_graph(n=30, m=100, seed=6)
    # graphs = [g1, g2, g3]

    print("Number graphs: ", len(graphs))
    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)

    # ========= DynGEM ===========
    if params.is_dyge:
        print("=============== DynGEM ============")
        # -------- Training ----------
        dy_ge, dy_embeddings = dyngem_alg(graphs=graphs, params=params)
        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")

    # ============== Node2Vec ============
    if params.is_node2vec:
        print("=============== Node2vec ============")
        dy_embeddings = node2vec_alg(
            graphs=graphs,
            embedding_dim=params.embedding_dim,
            folder_path=params.node2vec_emb_folder,
            is_load_emb=params.is_load_n2v_model
        )
        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")

    # == == == == == == == = SDNE == == == == == ==
    if params.is_sdne:
        print("=============== SDNE ============")
        dy_embeddings = sdne_alg(
            graphs=graphs,
            params=params
        )

        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")
