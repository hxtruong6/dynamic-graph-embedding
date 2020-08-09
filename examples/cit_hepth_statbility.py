import os
from os.path import exists
from time import time
import numpy as np
import networkx as nx
import warnings
from node2vec import Node2Vec

from src.data_preprocessing.graph_preprocessing import read_dynamic_graph
from src.utils.checkpoint_config import CheckpointConfig
from src.utils.stable_evaluate import stability_constant
from src.dyn_ge import TDynGE
from src.utils.graph_util import print_graph_stats, draw_graph

warnings.filterwarnings("ignore")


# If model_idx -> must be train from previous model
def train_overall(dy_ge: TDynGE, lr):
    '''
    Method to create saved model on disk for later optimizer
    :return:
    '''
    dy_ge.train(prop_size=prop_size, epochs=1, skip_print=1,
                net2net_applied=net2net_applied, learning_rate=lr,
                batch_size=batch_size, folder_path=weight_model_folder)


def train_model(dy_ge: TDynGE):
    print("\n-----------\nStart total training...")

    print("Overall training")
    train_overall(dy_ge, lr=learning_rate_list[0])

    print("\n### ==\tOptimize model training == ###")

    start_time_train = time()
    for model_idx in range(len(graphs)):
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


def train_model_at_index(dy_ge: TDynGE, ):
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


def check_current_loss_model(dy_ge: TDynGE):
    for model_idx in range(len(graphs)):
        dy_ge.train_at(model_index=model_idx, folder_path=weight_model_folder,
                       epochs=1, learning_rate=1e-7,
                       is_load_from_previous_model=False)


def dyn_gem_alg():
    # ----------- Training area -----------
    dy_ge = TDynGE(
        graphs=graphs, embedding_dim=embedding_dim,
        alpha=alpha, beta=beta, l1=l1, l2=l2
    )
    if is_just_load_model:
        print("\n-----------\nStart load model...")
        dy_ge.load_models(folder_path=weight_model_folder)
    elif specific_model_index is not None:
        train_model_at_index(dy_ge)
    else:
        train_model(dy_ge)

    # Uncomment to know current loss value
    check_current_loss_model(dy_ge)

    print("Saving embeddings...")
    dy_ge.save_embeddings(folder_path=embeddings_folder)

    print("Loading embedding...")
    # dy_embeddings = dy_ge.load_embeddings(folder_path=embeddings_folder)
    dy_embeddings = dy_ge.get_all_embeddings()
    # print(np.array_equal(dy_embeddings, dyn))

    return dy_ge, dy_embeddings


def node2vec_alg():
    dy_embeddings = []
    for g in graphs:
        node2vec = Node2Vec(g, dimensions=embedding_dim, walk_length=10, num_walks=200,
                            workers=4)  # Use temp_folder for big graphs
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embedding = [model[str(u)] for u in sorted(g.nodes)]
        dy_embeddings.append(np.array(embedding))
    dy_embeddings = np.array(dy_embeddings)
    return model, dy_embeddings


if __name__ == "__main__":
    # Algorithm
    is_dyngem = True
    is_node2vec = True

    # ----------- Run part --------------
    is_just_load_model = True  # Set True to run evaluate faster. If not: train->evaluate
    # ------------ Params -----------
    folder_data = "./data/cit_hepth"
    weight_model_folder = "./models/cit_hepth_stability"
    embeddings_folder = "./embeddings/cit_hepth_stability"

    epochs = 200
    skip_print = 200
    batch_size = 512  # 512
    early_stop = 100  # 100
    seed = 6
    prop_size = 0.3

    # empty if not need to train with list lr else priority
    learning_rate_list = [
        0.0005,
        # 3e-4, 1e-4,
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
    embedding_dim = 100
    net2net_applied = False
    checkpoint_config = CheckpointConfig(number_saved=50, folder_path=weight_model_folder + "_ck")

    # link prediction params
    show_acc_on_edge = True
    top_k = 10
    drop_node_percent = 0.2

    # ====================== Settings =================
    np.random.seed(seed=seed)

    if not exists(weight_model_folder):
        os.makedirs(weight_model_folder)

    if not exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    # ==================== Data =========================
    graphs, idx2node = read_dynamic_graph(
        folder_path=folder_data,
        limit=None,
        convert_to_idx=True
    )
    # g1 = nx.gnm_random_graph(n=10, m=15, seed=6)
    # g2 = nx.gnm_random_graph(n=15, m=30, seed=6)
    # g3 = nx.gnm_random_graph(n=30, m=100, seed=6)
    # graphs = [g1, g2, g3]

    # =============================================

    print("Number graphs: ", len(graphs))
    print("Origin graphs:")
    for i, g in enumerate(graphs):
        print_graph_stats(g, i, end="\t")
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")
        # draw_graph(g, limit_node=25)

    # ========= DynGEM ===========
    if is_dyngem:
        print("=============== DynGEM ============")
        # -------- Training ----------
        dy_ge, dy_embeddings = dyn_gem_alg()

        # -------- Stability constant -------------
        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")

    # ============== Node2Vec ============
    if is_node2vec:
        print("=============== Node2vec ============")
        _, dy_embeddings = node2vec_alg()
        print(f"Stability constant= {stability_constant(graphs=graphs, embeddings=dy_embeddings)}")
