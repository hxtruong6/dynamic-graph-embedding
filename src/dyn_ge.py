from os.path import join
import os
from time import time
import networkx as nx
from src.static_ge import StaticGE
from src.utils.autoencoder import Autoencoder
from src.utils.model_utils import get_hidden_layer, handle_expand_model, save_custom_model, load_custom_model
from src.utils.visualize import plot_embedding


class DynGE(object):
    def __init__(self, graphs, embedding_dim, v1=0.001, v2=0.001):
        super(DynGE, self).__init__()
        if not graphs:
            raise ValueError("Must be provide graphs data")

        self.graphs = graphs
        self.graph_len = len(graphs)
        self.embedding_dim = embedding_dim
        self.v1 = v1
        self.v2 = v2
        self.static_ges = []
        self.model_folder_paths = []

    def get_all_embeddings(self):
        return [ge.get_embedding() for ge in self.static_ges]

    def get_embedding(self, index):
        if index < 0 or index >= self.graph_len:
            raise ValueError("index is invalid!")
        return self.static_ges[index].get_embedding()

    def train(self, prop_size=0.4, batch_size=64, epochs=100, filepath="../models/generate/", skip_print=5,
              net2net_applied=False, learning_rate=0.003):
        init_hidden_dims = get_hidden_layer(prop_size=prop_size, input_dim=len(self.graphs[0].nodes()),
                                            embedding_dim=self.embedding_dim)
        model = Autoencoder(
            input_dim=len(self.graphs[0].nodes()),
            embedding_dim=self.embedding_dim,
            hidden_dims=init_hidden_dims,
            v1=self.v1,
            v2=self.v2
        )
        ge = StaticGE(G=self.graphs[0], model=model)

        self.model_folder_paths.append({
            "folder_path": join(filepath, "graph_0"),
            "name": "graph_0"
        })

        print(f"--- Training graph {0} ---")
        start_time = time()
        ge.train(batch_size=batch_size, epochs=epochs, skip_print=skip_print, learning_rate=learning_rate,
                 model_folder_path=self.model_folder_paths[0])
        print(f"Training time in {round(time() - start_time, 2)}s")

        # model.info(show_config=True)
        self.static_ges.append(ge)
        save_custom_model(model=model, model_folder_path=self.model_folder_paths[0])

        for i in range(1, len(self.graphs)):
            graph = nx.Graph(self.graphs[i])
            input_dim = len(graph.nodes())
            prev_model = self._create_prev_model(index=i - 1)
            curr_model = handle_expand_model(model=prev_model, input_dim=input_dim,
                                             prop_size=prop_size, net2net_applied=net2net_applied)
            ge = StaticGE(G=graph, model=curr_model)
            self.model_folder_paths.append({
                "folder_path": join(filepath, f"graph_{i}"),
                "name": f"graph_{i}"
            })

            print(f"--- Training graph {i} ---")
            start_time = time()
            ge.train(batch_size=batch_size, epochs=epochs, skip_print=skip_print, learning_rate=learning_rate,
                     model_folder_path=self.model_folder_paths[i])
            print(f"Training time in {round(time() - start_time, 2)}s")

            self.static_ges.append(ge)
            save_custom_model(model=ge.get_model(), model_folder_path=self.model_folder_paths[i])

    def _create_prev_model(self, index):
        model = load_custom_model(self.model_folder_paths[index])
        return model

    def load_models(self, folder_path):
        model_folders_paths = os.listdir(folder_path)
        self.model_folder_paths = []
        self.static_ges = []
        for i, folder in enumerate(model_folders_paths):
            model_folder_path = {
                'folder_path': join(folder_path, folder),
                'name': folder
            }
            model = load_custom_model(model_folder_path=model_folder_path)
            self.model_folder_paths.append(model_folder_path)

            ge = StaticGE(G=self.graphs[i], model=model)
            self.static_ges.append(ge)

    def train_from_checkpoint(self, folder_path, batch_size=64, epochs=10, skip_print=10, learning_rate=0.001,
                              filepath=None):
        self.load_models(folder_path=folder_path)
        if filepath is None:
            filepath = folder_path

        for i in range(self.graph_len):
            self.static_ges[i].train(batch_size=batch_size, epochs=epochs, skip_print=skip_print,
                                     learning_rate=learning_rate)

            model_folder_path = {
                "folder_path": join(filepath, f"graph_{i}"),
                "name": f"graph_{i}"
            }
            save_custom_model(model=self.static_ges[i].get_model(), model_folder_path=model_folder_path)


if __name__ == "__main__":
    g1 = nx.complete_graph(100)
    g2 = nx.complete_graph(150)
    g3 = nx.complete_graph(180)
    graphs = [g1, g1, g1]
    dy_ge = DynGE(graphs=graphs, embedding_dim=4)
    # dy_ge.train(prop_size=0.4, epochs=300, skip_print=30, net2net_applied=False, learning_rate=0.0005,
    #             filepath="../models/generate/")
    dy_ge.load_models(folder_path="../models/generate")
    # dy_ge.train_from_checkpoint(folder_path="../models/generate/", filepath="../models/checkpoints_1", epochs=200,
    #                             skip_print=20, learning_rate=0.00005)

    embeddings = dy_ge.get_all_embeddings()
    for e in embeddings:
        plot_embedding(embedding=e)
