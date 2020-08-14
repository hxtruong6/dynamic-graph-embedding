from time import time
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import lightgbm
import pandas as pd
import matplotlib.pyplot as plt


def get_unconnected_pairs_(G: nx.Graph, cutoff=2, n_limit=None):
    edges_len = dict(nx.all_pairs_shortest_path_length(G, cutoff=cutoff))

    unconnected_pairs = []
    appended_pairs = {}

    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    for u in nodes:
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
            if n_limit is not None and len(unconnected_pairs) >= n_limit:
                break

    return unconnected_pairs


def run_link_pred_evaluate(graph_df, embeddings, alg=None, num_boost_round=10000, early_stopping_rounds=100):
    if alg == "Node2Vec":
        x = [(embeddings[str(i)] + embeddings[str(j)]) for i, j in zip(graph_df['node_1'], graph_df['node_2'])]
    else:
        x = [(embeddings[i] + embeddings[j]) for i, j in zip(graph_df['node_1'], graph_df['node_2'])]

    # TODO: check unbalance dataset
    # TODO: cannot split with big data
    data_ = np.array(x)
    print("train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(
        data_,
        graph_df['link'],
        test_size=0.25,
        random_state=35,
        stratify=graph_df['link']
    )
    print(
        f"|Link=1|={list(graph_df['link']).count(1)}\t"
        f"|Link=0|={list(graph_df['link']).count(0)}\t\t"
        f"|Percent Link=1/Link=0|={round(list(graph_df['link']).count(1) / list(graph_df['link']).count(0), 4)}"
    )

    train_data = lightgbm.Dataset(X_train, y_train)
    test_data = lightgbm.Dataset(X_test, y_test)

    # TODO: add GPU training
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
def preprocessing_graph_for_link_prediction(G: nx.Graph, k_length=2, drop_node_percent=1, seed=6, edge_rate=None):
    np.random.seed(seed)
    print("Pre-processing graph for link prediction...")
    start_time = time()

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
        if G_partial.degree[u] < 2 or G_partial.degree[v] < 2:
            continue
        G_partial.remove_edge(u, v)
        if G_partial.number_of_nodes() != G.number_of_nodes():
            print("Delete wrong egde: {} {}".format(u, v))
            G_partial.add_edge(u, v)
        # TODO: Check connected component
        # if nx.number_connected_components(G_partial) == 1 and G_partial.number_of_nodes() == initial_nodes_len:
        omissible_links.append({
            'node_1': u,
            'node_2': v
        })
        dropped_node_count += 1

    assert G_partial.number_of_nodes() == G.number_of_nodes()

    # create data frame of removable edges
    removed_edge_graph_df = pd.DataFrame(data=omissible_links, columns=['node_1', 'node_2'])
    removed_edge_graph_df['link'] = 1  # add the target variable 'link'
    print(f"{round(time() - temp_time)}s")

    # print("Random drop some unconnected link")
    # print("Before drop length graph_df = ", len(graph_df))
    # remove_n = max(len(graph_df) - len(removed_edge_graph_df) * 9, 0)
    # drop_indices = np.random.choice(graph_df.index, remove_n, replace=False)
    # graph_df = graph_df.drop(drop_indices)
    # print("After drop length graph_df = ", len(graph_df))

    # Get possible edge can form in the future
    temp_time = time()
    print("\tGet possible unconnected link...", end=" ")
    if edge_rate is not None:
        expect_unconnected_links_len = len(removed_edge_graph_df) * (1 - edge_rate) / edge_rate
        all_unconnected_pairs = get_unconnected_pairs_(G, cutoff=k_length, n_limit=expect_unconnected_links_len)
    else:
        all_unconnected_pairs = get_unconnected_pairs_(G, cutoff=k_length)

    graph_df = pd.DataFrame(data=all_unconnected_pairs, columns=['node_1', 'node_2'])
    graph_df['link'] = 0  # add target variable 'link'
    print(f"{round(time() - temp_time)}s")

    print("Rate: ", len(removed_edge_graph_df) / len(graph_df))

    print("\tCreate data frame have potential links and removed link.")
    graph_df = graph_df.append(removed_edge_graph_df[['node_1', 'node_2', 'link']], ignore_index=True)
    graph_df = graph_df.astype(int)

    print(f"Processed graph in {round(time() - start_time, 2)}s")
    return graph_df, G_partial


def plot_link_prediction_graph(G: nx.Graph, pred_edges: [], pred_acc=None, idx2node=None):
    G = G.copy()
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
