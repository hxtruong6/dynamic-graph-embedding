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


def check_link_prediction(embedding, train_graph: nx.Graph, origin_graph: nx.Graph, k_query):
    def get_precisionK(max_index):
        print("\nGet Precision@K ...")
        similarity = get_similarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)[::-1]
        true_pred_count = 0
        K = 0
        prec_k_list = []
        node_size = train_graph.number_of_nodes()
        for ind in sortedInd:
            u = ind // node_size
            v = ind % node_size
            if u == v or not origin_graph.has_edge(u, v):
                continue

            K += 1
            if train_graph.has_edge(u, v):
                true_pred_count += 1

            prec_k_list.append(true_pred_count / K)
            if K > max_index:
                break
        return prec_k_list

    precisionK_list = get_precisionK(np.max(k_query))
    k_query_res = []
    for k in k_query:
        print(f"Precison@{k}={precisionK_list[k - 1]}")
        k_query_res.append(precisionK_list[k - 1])
    return k_query_res
