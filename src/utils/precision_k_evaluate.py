import numpy as np
import networkx as nx


def get_similarity(result):
    return np.dot(result, result.T)


def dyge_reconstruction_evaluation(graphs, dy_embeddings,
                                   k_query=None):
    if k_query is None:
        k_query = [1, 2, 10, 20, 100, 200, 1000, 2000, 4000, 6000, 8000, 10000]
    sum_AP = 0
    for i, g in enumerate(graphs):
        reconstruction_prec = reconstruction_precision_k(embedding=dy_embeddings[i], graph=g, k_query=k_query)
        print(reconstruction_prec)
        sum_AP += reconstruction_prec[1]
    print("mAP: ", sum_AP / len(graphs))


def reconstruction_precision_k(embedding, graph: nx.Graph,
                               k_query=None):
    if k_query is None:
        k_query = [1, 2, 10, 20, 100, 200, 1000, 2000, 4000, 6000, 8000, 10000]

    def get_precisionK(max_index):
        similarity = get_similarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)[::-1]
        K = 0
        true_pred_count = 0
        prec_k_list = []
        node_size = graph.number_of_nodes()

        true_size_edges = graph.number_of_edges()
        precision_list = []

        for ind in sortedInd:
            u = ind // node_size
            v = ind % node_size
            if u == v:
                continue

            K += 1
            if graph.has_edge(u, v):
                true_pred_count += 1
                if true_pred_count <= true_size_edges:
                    precision_list.append(true_pred_count / K)

            if K <= max_index:
                prec_k_list.append(true_pred_count / K)

            if true_pred_count > true_size_edges and K > max_index:
                break

        AP = np.sum(precision_list) / true_size_edges
        return prec_k_list, AP

    print('\nReconstruction Precision K')
    precisionK_list, AP = get_precisionK(np.max(k_query))
    k_query_res = []
    for k in k_query:
        print(f"Precison@K({k})=\t{precisionK_list[k - 1]}")
        k_query_res.append(precisionK_list[k - 1])
    return k_query_res, AP


def check_link_predictionK(embedding, train_graph: nx.Graph, origin_graph: nx.Graph, k_query):
    def get_precisionK(max_index):
        print("\nGet Precision@K ...")
        similarity = get_similarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)[::-1]
        true_pred_count = 0
        K = 0
        prec_k_list = []
        node_size = train_graph.number_of_nodes()

        removed_links_size = origin_graph.number_of_edges() - train_graph.number_of_edges()
        precision_list = []

        for ind in sortedInd:
            u = ind // node_size
            v = ind % node_size
            # Only check removed link from origin graph.
            if u == v or train_graph.has_edge(u, v):
                continue
            # print(similarity[ind])

            K += 1
            # This mean train_graph can be predict u and v has a link.
            if origin_graph.has_edge(u, v):
                true_pred_count += 1
                if true_pred_count <= removed_links_size:
                    precision_list.append(true_pred_count / K)

            if K <= max_index:
                prec_k_list.append(true_pred_count / K)

            if true_pred_count > removed_links_size and K > max_index:
                break
        # Due to only one query Q = 1
        AP = np.sum(precision_list) / removed_links_size
        return prec_k_list, AP

    precisionK_list, AP = get_precisionK(np.max(k_query))
    k_query_res = []
    for k in k_query:
        print(f"Precison@{k}={precisionK_list[k - 1]}")
        k_query_res.append(precisionK_list[k - 1])
    return k_query_res, AP
