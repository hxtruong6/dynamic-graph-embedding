import numpy as np
import networkx as nx
import os
from os import listdir
from os.path import isfile, join, exists
import re
import matplotlib.pyplot as plt

from time import time

from examples.fb_wosn_friends import load_graphs_df, load_processed_graphs, save_graph_df, save_processed_graph
from src.dyn_ge import DynGE

# Dataset link: https://snap.stanford.edu/data/cit-HepTh.html
from src.static_ge import StaticGE
from src.utils.graph_util import print_graph_stats, graph_to_graph_idx, draw_graph
from src.utils.link_prediction import preprocessing_graph_for_link_prediction, run_link_pred_evaluate, run_predict, \
    top_k_prediction_edges


def handle_citHepTH_dataset(edge_list_path=None, abstract_path=None, verbose=False):
    if edge_list_path is None or abstract_path is None:
        raise ValueError("Must be provide path of dataset")

    print("Reading Cit-HepTH dataset...")
    begin_time = time()
    G = nx.read_edgelist(edge_list_path, nodetype=int)
    V = G.nodes()

    year_folder = os.listdir(abstract_path)
    abs_nodes_dic = []
    abs_nodes = []
    nodes_by_year = {}
    for y in sorted(year_folder):
        nodes_by_year[y] = []
        curr_path = join(abstract_path, y)
        files = [f for f in listdir(curr_path) if isfile(join(curr_path, f))]
        for file in files:
            v = int(file.strip().split('.')[0])
            if v not in V:
                continue
            abs_nodes.append(v)
            nodes_by_year[y].append(v)
            # file format: '9205018.abs'
            # with open(join(curr_path, file)) as fi:
            #     content = fi.read()
            #     contents = re.split(r'(\\)+', content)
            #     abs_nodes_dic.append({v: {"info": contents[2], "abstract": contents[4]}})
        # print(f"Year {y}: number of nodes: {len(nodes_by_year[y])}")

    graphs = []
    years = list(nodes_by_year.keys())
    prev_year = None
    for i, year in enumerate(reversed(years)):
        graph = None
        if i == 0:
            graph = G.copy()
        else:
            graph = graphs[i - 1].copy()
            graph.remove_nodes_from(nodes_by_year[prev_year])
        if verbose:
            print(f"Year {year}: |V|={len(graph.nodes())}\t |E|={len(graph.edges())}")
        prev_year = year
        graphs.append(graph)

    print(f"Reading in {round(time() - begin_time, 2)}s. Done!")
    return list(reversed(graphs))

def get_ciHepTH_dataset():
    return handle_citHepTH_dataset(
        edge_list_path="../data/cit-HepTh/cit-HepTh.txt",
        abstract_path="../data/cit-HepTh/cit-HepTh-abstracts/"
    )


def ciHepTH_link_prediction(G: nx.Graph, epochs=10, top_k=10, show_acc_on_edge=False):
    # unconnected_egdes =x
    G_df, G_partial = preprocessing_graph_for_link_prediction(G=G)
    # TODO: check remove maximum 15% omissible egde in total egdes.
    ge = StaticGE(G=G_partial, embedding_dim=4, hidden_dims=[8])
    ge.train(epochs=10, skip_print=20, learning_rate=0.001)
    embedding = ge.get_embedding()

    link_pred_model = run_link_pred_evaluate(G_df, embedding, num_boost_round=2000)
    possible_edges_df = G_df[G_df['link'] == 0]
    y_pred = run_predict(data=possible_edges_df, embedding=embedding, model=link_pred_model)
    top_k_edges = top_k_prediction_edges(G=G, y_pred=y_pred, possible_edges_df=possible_edges_df, top_k=top_k,
                                         show_acc_on_edge=show_acc_on_edge)
    return top_k_edges


if __name__ == "__main__":
    # ------------ Params -----------
    folder_data = "../data/cit-HepTh"
    processed_data_folder = "../processed_data/cit-HepTh"
    weight_model_folder = "../models/synthetic"
    load_model = False
    load_processed_data = False
    epochs = 500
    skip_print = 100
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
    original_graphs = handle_citHepTH_dataset(
        edge_list_path="../data/cit-HepTh/cit-HepTh.txt",
        abstract_path="../data/cit-HepTh/cit-HepTh-abstracts/",
        verbose=False
    )
    original_graphs = original_graphs[:3]

    print("Number graphs: ", len(original_graphs))
    print("Origin graphs:")
    for i, g in enumerate(original_graphs):
        print_graph_stats(g, i)
        print(f"Isolate nodes: {nx.number_of_isolates(g)}")

    graphs2idx = []
    idx2nodes = []
    for g in original_graphs:
        graph2idx, idx2node = graph_to_graph_idx(g)
        graphs2idx.append(graph2idx)
        idx2nodes.append(idx2node)
    graphs = graphs2idx

    draw_graph(graphs[0], idx2node=idx2nodes[0])

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

        print(f"[ALL] Processed in {round(time() - start_time, 2)}s\n")

    draw_graph(g=G_partial_list[0], pos=nx.spring_layout(graphs[0], seed=6))

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
        dy_ge.train(prop_size=0.3, epochs=epochs, skip_print=skip_print, net2net_applied=False, learning_rate=0.0001,
                    batch_size=batch_size, folder_path=weight_model_folder, save_model_point=None)
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
                                             limit_node=50, idx2node=idx2nodes[i])
