import networkx as nx
import warnings

from src.data_preprocessing.data_preprocessing import read_dynamic_graph

warnings.filterwarnings("ignore")

from src.dyn_ge import DynGE
from src.utils.graph_util import graph_to_graph_idx, print_graph_stats, draw_graph
from src.utils.link_prediction import preprocessing_graph_for_link_prediction, run_evaluate, run_predict, \
    top_k_prediction_edges

if __name__ == "__main__":
    # folder_data = "../data/as-733"
    # original_graphs = read_dynamic_graph(folder_path="../data/as-733", limit=2)
    g1 = nx.gnm_random_graph(n=40, m=200, seed=6)
    original_graphs = [g1]

    print("Number graphs: ", len(original_graphs))
    print("Origin graphs:")
    for i, g in enumerate(original_graphs):
        print_graph_stats(g, i)

    graphs2idx = []
    idx2nodes = []
    for g in original_graphs:
        graph2idx, idx2node = graph_to_graph_idx(g)
        graphs2idx.append(graph2idx)
        idx2nodes.append(idx2node)

    graphs = graphs2idx
    # draw_graph(g1, pos=nx.spring_layout(graphs[0], seed=6))
    # draw_graph(graphs[0], idx2node=idx2nodes[0])

    G_dfs = []
    G_partial_list = []
    for g in graphs:
        g_df, g_partial = preprocessing_graph_for_link_prediction(G=g, k_length=2, drop_node_percent=0.2)
        G_dfs.append(g_df)
        G_partial_list.append(g_partial)

    # draw_graph(g=G_partial_list[0], pos=nx.spring_layout(graphs[0], seed=6))

    print("After processing for link prediction graphs:")
    for i, g in enumerate(G_partial_list):
        print_graph_stats(g, i)
    # ------------ Params -----------
    load_model = False
    folder_path = "../models/fb"
    epochs = 100
    skip_print = 100

    # link prediction params
    show_acc_on_edge = True
    top_k = 10

    # -------------------------------
    dy_ge = DynGE(graphs=G_partial_list, embedding_dim=4)
    if load_model and folder_path:
        dy_ge.load_models(folder_path=folder_path)
    else:
        dy_ge.train(prop_size=0.3, epochs=epochs, skip_print=skip_print, net2net_applied=False, learning_rate=0.0005,
                    filepath="../models/synthetic/")

    dy_embeddings = dy_ge.get_all_embeddings()

    # ----- run evaluate link prediction -------
    for i in range(len(graphs)):
        G_df = G_dfs[i]
        link_pred_model = run_evaluate(data=G_df, embedding=dy_embeddings[i], num_boost_round=10000)
        possible_edges_df = G_df[G_df['link'] == 0]
        y_pred = run_predict(data=possible_edges_df, embedding=dy_embeddings[i], model=link_pred_model)
        top_k_edges = top_k_prediction_edges(G=graphs[i], y_pred=y_pred, possible_edges_df=possible_edges_df,
                                             top_k=top_k, show_acc_on_edge=show_acc_on_edge, plot_link_pred=True,
                                             limit_node=50, idx2node=idx2nodes[i])
