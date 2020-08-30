import warnings

import networkx as nx

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph
from src.utils.config_file import read_config_file
from src.utils.data_utils import load_dy_embeddings, load_node2vec_embeddings
from src.utils.graph_util import print_graph_stats
from src.utils.setting_param import SettingParam
from src.utils.stable_evaluate import stability_constant

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    # dataset_name = "soc_wiki"
    # params = {
    #     # 'algorithm': {
    #     'is_dyge': True,
    #     'is_node2vec': True,
    #     'is_sdne': True,
    #
    #     # 'folder_paths': {
    #     'dataset_folder': f"./data/{dataset_name}",
    #     'processed_link_pred_data_folder': f"./saved_data/processed_data/{dataset_name}_stability",
    #
    #     'dyge_emb_folder': f"./saved_data/embeddings/{dataset_name}_stability",
    #     'node2vec_emb_folder': f"./saved_data/node2vec_emb/{dataset_name}_stability",
    #     'sdne_emb_folder': f"./saved_data/sdne_emb/{dataset_name}_stability",
    # }
    # params = SettingParam(**params)
    params = read_config_file(filepath="./stability_configuration.ini", config_task="stability")
    is_mAP_eval = False
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

    # ========= DynGEM ===========
    if params.is_dyge:
        print("=============== DynGEM ============")
        # -------- Training ----------
        dy_embeddings = load_dy_embeddings(params.dyge_emb_folder)
        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")

    # ============== Node2Vec ============
    if params.is_node2vec:
        print("=============== Node2vec ============")
        dy_embeddings = load_node2vec_embeddings(graphs, folder_path=params.node2vec_emb_folder)
        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")

    # ==================== SDNE ===============
    if params.is_sdne:
        print("=============== SDNE ============")
        dy_embeddings = load_dy_embeddings(folder_path=params.sdne_emb_folder)
        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")
