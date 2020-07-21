from time import time
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import lightgbm
import pandas as pd
import matplotlib.pyplot as plt


def get_unconnected_pairs(G: nx.Graph):
    # TODO: convert to sparse matrix
    node_list = list(G.nodes())
    adj_G = nx.adj_matrix(G)

    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix. find all unconnected node with maximum 2nd order
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset, adj_G.shape[1]):
            if i != j:
                if nx.shortest_path_length(G, i, j) <= 2:
                    if adj_G[i, j] == 0:
                        all_unconnected_pairs.append([node_list[i], node_list[j]])

        offset = offset + 1

    return all_unconnected_pairs


def get_unconnected_pairs_(G: nx.Graph, k_length=2):
    edges_len = dict(nx.all_pairs_shortest_path_length(G, cutoff=k_length))

    unconnected_pairs = []
    appended_pairs = {}
    for u in G.nodes():
        if u not in appended_pairs:
            appended_pairs[u] = {}

        for v in edges_len[u].keys():
            if v not in appended_pairs:
                appended_pairs[v] = {}
            if u != v and v not in appended_pairs[u] and not G.has_edge(u, v):
                unconnected_pairs.append({
                    'node_1': u,
                    'node_2': v
                })

                appended_pairs[u][v] = True
                appended_pairs[v][u] = True

    return unconnected_pairs


def run_link_pred_evaluate(data, embedding, alg=None, num_boost_round=10000, early_stopping_rounds=100):
    print("--> Run link predict evaluation ---")
    if alg == "Node2Vec":
        x = [(embedding[str(i)] + embedding[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]
    else:
        x = [(embedding[i] + embedding[j]) for i, j in zip(data['node_1'], data['node_2'])]

    # TODO: check unbalance dataset
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(x),
        data['link'],
        test_size=0.25,
        random_state=35,
        stratify=data['link']
    )
    print(
        f"|Link=1|={list(data['link']).count(1)}\t"
        f"|Link=0|={list(data['link']).count(0)}\t\t"
        f"|Percent Link=1/Link=0|={round(list(data['link']).count(1) / list(data['link']).count(0), 4)}"
    )

    train_data = lightgbm.Dataset(X_train, y_train)
    test_data = lightgbm.Dataset(X_test, y_test)

    # define parameters
    parameters = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'num_threads': 4,
        'seed': 6,
        'verbosity': -1
    }

    # train lightGBM model
    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=test_data,
                           num_boost_round=num_boost_round,
                           early_stopping_rounds=early_stopping_rounds,
                           verbose_eval=100,
                           )

    y_pred = model.predict(X_test)

    try:
        print(f"#----\nROC AUC Score: {round(roc_auc_score(y_test, y_pred), 2)}")
    except ValueError:
        print("ROC AUC has only one class: ", int(y_pred[0]))
    # roc_curve(y_test, y_pred)
    return model


def top_k_prediction_edges(G: nx.Graph, y_pred, possible_edges_df, top_k, show_acc_on_edge=False, plot_link_pred=False,
                           limit_node=100, idx2node=None):
    # get top K link prediction
    # sorted_y_pred, sorted_possible_edges = zip(*sorted(zip(y_pred, possible_egdes)))
    node_1 = possible_edges_df['node_1'].to_list()
    node_2 = possible_edges_df['node_2'].to_list()

    # unconnected_edges = [possible_egdes_df['node_1'].to_list(), possible_egdes_df['node_2'].to_list()]
    unconnected_edges = [(node_1[i], node_2[i]) for i in range(len(node_1))]
    sorted_y_pred, sorted_possible_edges = (list(t) for t in zip(*sorted(zip(y_pred, unconnected_edges), reverse=True)))

    if plot_link_pred:
        if len(G.nodes) > limit_node:
            G.remove_nodes_from(nodes=list(G.nodes)[limit_node:])
        if show_acc_on_edge:
            plot_link_prediction_graph(G=G, pred_edges=sorted_possible_edges[:top_k], pred_acc=sorted_y_pred[:top_k],
                                       idx2node=idx2node)
        else:
            plot_link_prediction_graph(G=G, pred_edges=sorted_possible_edges[:top_k], idx2node=idx2node)

    if idx2node is None:
        original_pred_edges = sorted_possible_edges[:top_k]
    else:
        original_pred_edges = [(idx2node[u], idx2node[v]) for u, v in sorted_possible_edges[:top_k]]

    print(f"Top {top_k} predicted edges: edge|accuracy")
    for i in range(top_k):
        print(f"{original_pred_edges[i]} : {round(sorted_y_pred[i], 2)}")

    return original_pred_edges, sorted_y_pred[:top_k]


def run_predict(data, embedding, model):
    x = [(embedding[i] + embedding[j]) for i, j in zip(data['node_1'], data['node_2'])]
    y_pred = model.predict(x)
    return y_pred


# https://www.analyticsvidhya.com/blog/2020/01/link-prediction-how-to-predict-your-future-connections-on-facebook/
def preprocessing_graph_for_link_prediction(G: nx.Graph, k_length=2, drop_node_percent=1):
    print("Pre-processing graph for link prediction...")
    start_time = time()

    # Get possible edge can form in the future
    print("\tGet possible unconnected link...", end=" ")
    temp_time = time()

    all_unconnected_pairs = get_unconnected_pairs_(G, k_length=k_length)
    data = pd.DataFrame(data=all_unconnected_pairs, columns=['node_1', 'node_2'])
    data['link'] = 0  # add target variable 'link'
    print(f"{round(time() - temp_time)}s")

    # Drop some edges which not make graph isolate
    print("\tDrop some current links...", end=" ")
    temp_time = time()

    initial_nodes_len = G.number_of_nodes()
    initial_edges_len = G.number_of_edges()
    dropped_node_count = 0
    G_partial = G.copy()
    omissible_links = []
    initial_edges = list(G.edges)
    np.random.shuffle(initial_edges)

    for u, v in tqdm(initial_edges):
        if (dropped_node_count / initial_edges_len) >= drop_node_percent:
            break
        G_partial.remove_edge(u, v)
        if nx.number_connected_components(G_partial) == 1 and G_partial.number_of_nodes() == initial_nodes_len:
            omissible_links.append({
                'node_1': u,
                'node_2': v
            })
            dropped_node_count += 1
        else:
            G_partial.add_edge(u, v)

    # create dataframe of removable edges
    removed_edge_graph_df = pd.DataFrame(data=omissible_links, columns=['node_1', 'node_2'])
    removed_edge_graph_df['link'] = 1  # add the target variable 'link'
    print(f"{round(time() - temp_time)}s")

    print("\tCreate data frame have potential links and removed link.")
    data = data.append(removed_edge_graph_df[['node_1', 'node_2', 'link']], ignore_index=True)
    data = data.astype(int)

    print(f"Processed graph in {round(time() - start_time, 2)}s")
    return data, G_partial


def plot_link_prediction_graph(G: nx.Graph, pred_edges: [], pred_acc=None, idx2node=None):
    if pred_acc is None:
        pred_acc = []
    pos = nx.spring_layout(G, seed=6)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    pred_edges_ = []
    edges_labels = {}

    for i, (u, v) in enumerate(pred_edges):
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v)
            pred_edges_.append((u, v))
            if pred_acc:
                edges_labels[(u, v)] = round(pred_acc[i], 2)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=pred_edges_,
        # width=1,
        # alpha=0.5,
        edge_color='r'
    )

    if pred_acc:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels, font_color='red')

    if idx2node is not None:
        labels = {}
        for u in G.nodes:
            labels[u] = str(idx2node[u])
        nx.draw_networkx_labels(G, pos, labels=labels)
    else:
        nx.draw_networkx_labels(G, pos=pos)

    plt.axis('off')
    plt.show()
