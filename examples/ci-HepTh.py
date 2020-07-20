import numpy as np
import networkx as nx
import os
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt

from time import time
from src.dyn_ge import DynGE

# Dataset link: https://snap.stanford.edu/data/cit-HepTh.html
from src.static_ge import StaticGE
from src.utils.link_prediction import preprocessing_graph_for_link_prediction, run_evaluate, run_predict, \
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

    link_pred_model = run_evaluate(G_df, embedding, num_boost_round=2000)
    possible_edges_df = G_df[G_df['link'] == 0]
    y_pred = run_predict(data=possible_edges_df, embedding=embedding, model=link_pred_model)
    top_k_edges = top_k_prediction_edges(G=G, y_pred=y_pred, possible_edges_df=possible_edges_df, top_k=top_k,
                                         show_acc_on_edge=show_acc_on_edge)
    return top_k_edges


if __name__ == "__main__":
    graphs = handle_citHepTH_dataset(
        edge_list_path="../data/cit-HepTh/cit-HepTh.txt",
        abstract_path="../data/cit-HepTh/cit-HepTh-abstracts/",
        verbose=False
    )

    print("Number graphs: ", len(graphs))
    # graphs = graphs[:2]
    #
    # dy_ge = DynGE(graphs=graphs, embedding_dim=4, init_hidden_dims=[64, 16])
    # dy_ge.train()
    # embeddings = dy_ge.get_all_embeddings()
    # for e in embeddings:
    #     print(embeddings[:5])

    # g1 = nx.gnm_random_graph(n=40, m=200, seed=6)
    # nx.draw(g1, pos=nx.spring_layout(g1, seed=6))
    # nx.draw_networkx_labels(g1, pos=nx.spring_layout(g1, seed=6))
    # plt.show()
    #
    # g2 = nx.gnm_random_graph(n=60, m=400, seed=6)
    # nx.draw(g2, pos=nx.spring_layout(g2, seed=6))
    # nx.draw_networkx_labels(g2, pos=nx.spring_layout(g2, seed=6))
    # plt.show()
    #
    # graphs = [g1, g2]
    G_dfs = []
    G_partial_list = []
    for g in graphs:
        g_df, g_partial = preprocessing_graph_for_link_prediction(G=g)
        G_dfs.append(g_df)
        G_partial_list.append(g_partial)

    # ------------ Params -----------
    load_model = False
    folder_path = "../models/ci-HepTH"
    epochs = 1000
    skip_print = 100

    # link prediction params
    show_acc_on_edge = False
    top_k = 10

    # -------------------------------
    dy_ge = DynGE(graphs=G_partial_list, embedding_dim=4)
    if load_model and folder_path:
        dy_ge.load_models(folder_path=folder_path)
    else:
        dy_ge.train(prop_size=0.3, epochs=epochs, skip_print=skip_print, net2net_applied=False, learning_rate=0.0005,
                    filepath="../models/ci-HepTH/")

    dy_embeddings = dy_ge.get_all_embeddings()

    # ----- run evaluate link prediction -------
    for i in range(len(graphs)):
        G_df = G_dfs[i]
        link_pred_model = run_evaluate(data=G_df, embedding=dy_embeddings[i], num_boost_round=10000)
        possible_edges_df = G_df[G_df['link'] == 0]
        y_pred = run_predict(data=possible_edges_df, embedding=dy_embeddings[i], model=link_pred_model)
        top_k_edges = top_k_prediction_edges(G=graphs[i], y_pred=y_pred, possible_edges_df=possible_edges_df,
                                             top_k=top_k,
                                             show_acc_on_edge=show_acc_on_edge)
