import networkx as nx
import matplotlib.pyplot as plt


def print_graph_stats(G: nx.Graph, name=""):
    print(f"Graph {name}: |V|={G.number_of_nodes()}\t |E|={G.number_of_edges()}")


def preprocess_graph(graph):
    '''

    :param graph: graph input - adjacency matrix
    :return: idx2node, node2idx
    '''
    graph = nx.Graph(graph)
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def graph_to_graph_idx(g: nx.Graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in sorted(g.nodes):
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1

    graph2idx = nx.Graph()

    for u, v in g.edges():

        graph2idx.add_edge(node2idx[u], node2idx[v])

    return graph2idx, idx2node


def draw_graph(g: nx.Graph, seed=6, pos=None, idx2node=None):
    if pos is None:
        pos = nx.spring_layout(g, seed=seed)

    nx.draw(g, pos=pos)
    if idx2node is not None:
        labels = {}
        for u in g.nodes:
            labels[u] = str(idx2node[u])
        nx.draw_networkx_labels(g, pos, labels=labels)
    else:
        nx.draw_networkx_labels(g, pos=pos)

    plt.show()
