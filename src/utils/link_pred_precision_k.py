import numpy as np
import networkx as nx


def get_similarity(result):
    return np.dot(result, result.T)


def check_reconstruction(embedding, graph: nx.Graph, k_query):
    def get_precisionK(max_index):
        print("Get Precision@K...")
        similarity = get_similarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)[::-1]
        K = 0
        true_pred_count = 0
        prec_k_list = []
        node_size = graph.number_of_nodes()
        for ind in sortedInd:
            u = ind // node_size
            v = ind % node_size
            print(f"{u} - {v}| Link: {graph.has_edge(u, v)} | Similarity: {similarity[u][v]}")
            if u == v:
                continue
            if graph.has_edge(u, v):
                true_pred_count += 1
            K += 1

            prec_k_list.append(true_pred_count / K)
            if K > max_index:
                break
        return prec_k_list

    precisionK_list = get_precisionK(np.max(k_query))
    k_query_res = []
    for k in k_query:
        print(f"Precison@K({k})={precisionK_list[k - 1]}")
        k_query_res.append(precisionK_list[k - 1])
    return k_query_res


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
