import warnings
from time import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph
from src.static_ge import TStaticGE
from src.utils.checkpoint_config import CheckpointConfig
from src.utils.config_file import read_config_file
from src.utils.link_pred_precision_k import check_link_predictionK
from src.utils.model_training_utils import create_folder, dyngem_alg, link_pred_eva, node2vec_alg, sdne_alg
from src.utils.data_utils import save_processed_data, load_single_processed_data
from src.utils.graph_util import print_graph_stats
from src.utils.link_prediction import preprocessing_graph_for_link_prediction
from src.utils.model_utils import get_hidden_layer
from src.utils.setting_param import SettingParam

warnings.filterwarnings("ignore")
# sns.set('whitegrid')

if __name__ == "__main__":
    def _try_train(graph: nx.Graph, alpha, beta, emb_dim):
        hidden_dims = get_hidden_layer(
            prop_size=params.prop_size,
            input_dim=graph.number_of_nodes(),
            embedding_dim=emb_dim
        )
        ge = TStaticGE(G=graph, embedding_dim=emb_dim, hidden_dims=hidden_dims, l2=params.l2,
                       alpha=alpha, beta=beta, activation=params.sdne_activation)
        print("\n====\nConfiguration:")
        print(f"Emb_dim={emb_dim}\t|alpha={alpha}\t|beta={beta}")
        for lr in params.learning_rate_list:
            print(f"\tLr={lr}")
            ge.train(
                batch_size=None, epochs=params.epochs, skip_print=params.skip_print,
                learning_rate=lr, early_stop=params.early_stop,
                plot_loss=True, shuffle=params.sdne_shuffle
            )
        _, mAP = check_link_predictionK(embedding=ge.get_embedding(), train_graph=g_hidden_partial,
                                        origin_graph=graph,
                                        k_query=[2, 10, 100, 200, 1000, 10000])
        print(f"mAP = {mAP}")
        return round(mAP, 4)


    dataset_name = "tune_data"
    params = {
        # 'folder_paths': {
        'dataset_folder': f"./data/{dataset_name}",
        'processed_link_pred_data_folder': f"./saved_data/processed_data/{dataset_name}",

        'global_seed': 6,
        # Processed data
        'is_load_link_pred_data': True,

        # 'dyge_config': {
        'prop_size': 0.3,
        'epochs': 2,
        'skip_print': 50,
        'early_stop': 100,
        'learning_rate_list': [
            1e-3,
            # 1e-4,
            # 1e-5,
            # 1e-6
        ],
        'l2': 0.0005,

        # SDNE
        'sdne_shuffle': False,
        'sdne_load_model': False,
        'sdne_resume_training': False,
        'sdne_activation': 'relu',

        # 'link_pred_config': {
        'drop_node_percent': 0.2,
    }
    params = SettingParam(**params)
    params.show_loss = True

    # ====================== Settings =================
    np.random.seed(seed=params.global_seed)

    create_folder(params.processed_link_pred_data_folder)
    # ==================== Data =========================

    graphs, idx2node = read_dynamic_graph(
        folder_path=params.dataset_folder,
        limit=None,
        convert_to_idx=True
    )

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
            edge_rate=0.001
        )
        # NOTE: save idx graph. Not original graph
        save_processed_data(g_hidden_df, g_hidden_partial, folder=params.processed_link_pred_data_folder,
                            index=len(graphs) - 1)
        # draw_graph(g=g_partial, limit_node=25)
        print(f"[ALL] Processed in {round(time() - start_time, 2)}s\n")

    # Set last graph in dynamic graph is hidden graph
    origin_graph = graphs[-1]
    graphs[-1] = g_hidden_partial
    print("After processing for link prediction graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    # ========== SDNE ============
    emb_dims = [100, 120]
    alphas = [0.2, 0.4]
    betas = [10, 16]

    df = []
    for emb_dim in emb_dims:
        for alpha in alphas:
            for beta in betas:
                mAP = _try_train(graph=origin_graph, emb_dim=emb_dim, alpha=alpha, beta=beta)
                df.append([emb_dim, alpha, beta, mAP])

    df = pd.DataFrame(data=df, columns=['emb_dim', 'alpha', 'beta', 'mAP'])
    print(df.sort_values(by=['mAP'], ascending=False))

    # sns.set(style="whitegrid")
    fig_dims = (10, 65)
    fig, axs = plt.subplots(nrows=6, figsize=fig_dims)
    sns.lineplot(x='emb_dim', y='mAP', hue='alpha', data=df, ax=axs[0], marker="o")
    sns.lineplot(x='emb_dim', y='mAP', hue='beta', data=df, ax=axs[1], marker="o")
    sns.lineplot(x='alpha', y='mAP', hue='emb_dim', data=df, ax=axs[2], marker="o")
    sns.lineplot(x='alpha', y='mAP', hue='beta', data=df, ax=axs[3], marker="o")
    sns.lineplot(x='beta', y='mAP', hue='alpha', data=df, ax=axs[4], marker="o")
    sns.lineplot(x='beta', y='mAP', hue='emb_dim', data=df, ax=axs[5], marker="o")

    plt.savefig("tune.png")
    plt.show()
