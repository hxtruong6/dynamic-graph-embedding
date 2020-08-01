import networkx as nx
import numpy as np

from src.dyn_ge import TDynGE


def relative_stability(g1: nx.Graph, g2: nx.Graph, e1, e2):
    a1 = np.array(nx.adjacency_matrix(g1).todense())
    a2 = np.array(nx.adjacency_matrix(g2).todense())

    g1_length = g1.number_of_nodes()

    sub_node_set = set(g1.nodes).intersection(g2.nodes)

    f_numerator = 0
    s_numerator = 0
    s_denominator = 0
    f_denominator = 0
    for node in sub_node_set:
        f_numerator += np.sum(np.square(e2[node] - e1[node]))
        s_numerator += np.sum(np.square(a2[node][:g1_length] - a1[node]))
        f_denominator += np.square(e1[node])
        s_denominator += np.square(a1[node])

    f_numerator = np.sqrt(np.sum(f_numerator))
    s_numerator = np.sqrt(np.sum(s_numerator))
    f_denominator = np.sqrt(np.sum(f_denominator))
    s_denominator = np.sqrt(np.sum(s_denominator))

    res = (f_numerator / f_denominator) / (s_numerator / s_denominator)
    return res


def stability_constant(graphs: [nx.Graph], embeddings: []) -> float:
    stab_constant = 1e-6
    for i in range(len(graphs) - 1):
        rel_stability = relative_stability(g1=graphs[i], g2=graphs[i + 1], e1=embeddings[i], e2=embeddings[i + 1])
        print(f"[{i + 1} -> {i}] Relative_stab: ", rel_stability)
        stab_constant = max(stab_constant, rel_stability)

    return stab_constant


if __name__ == "__main__":
    g1 = nx.gnm_random_graph(n=5, m=6, seed=6)
    g2 = nx.gnm_random_graph(n=7, m=10, seed=6)
    g3 = nx.gnm_random_graph(n=15, m=25, seed=6)
    g4 = nx.gnm_random_graph(n=20, m=40, seed=6)
    graphs = [g1, g2, g3, g4]

    dy_ge = TDynGE(graphs=graphs, embedding_dim=2)
    dy_ge.train(prop_size=0.3, epochs=4000, skip_print=400,
                learning_rate=0.001, folder_path="../../models/synthetic")

    dyn_embeddings = dy_ge.get_all_embeddings()

    print(f"Stability constant= {stability_constant(graphs, embeddings=dyn_embeddings)}")
