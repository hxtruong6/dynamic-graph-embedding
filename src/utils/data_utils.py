from os import listdir, makedirs
from os.path import isfile, join, exists
import pandas as pd
import networkx as nx

from src.data_preprocessing.graph_preprocessing import get_graph_from_file


def load_processed_data(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    graphs_df = []
    graphs = []
    for f in sorted(files):
        if ".json" in f:
            graphs_df.append(pd.read_json(join(folder, f)))
        else:  # ".edgelist" in f is True
            graphs.append(get_graph_from_file(join(folder, f)))
    return graphs_df, graphs


def save_processed_data(graph_df: pd.DataFrame, graph: nx.Graph, folder, index):
    if not exists(folder):
        makedirs(folder)
    nx.write_edgelist(graph, f'{folder}/graph{index}.edgelist', data=False)
    graph_df.to_json(join(folder, f"graph_{index}.json"))
