import torch.nn as nn


class SettingParam(object):

    def __init__(self, **kwargs):
        self.global_seed = None
        self.show_loss = None
        self.dataset_name = None

        # algorithms
        self.is_dyge = None
        self.is_node2vec = None
        self.is_sdne = None

        #  folder_paths
        self.dataset_folder = None
        self.processed_link_pred_data_folder = None
        self.dyge_weight_folder = None
        self.dyge_emb_folder = None

        self.node2vec_emb_folder = None

        self.sdne_weight_folder = None
        self.sdne_emb_folder = None

        #  training_config
        self.is_load_link_pred_data = None
        self.is_load_dyge_model = None
        self.specific_dyge_model_index = None

        #  dyge_config
        self.prop_size = None
        self.embedding_dim = None
        self.epochs = None
        self.skip_print = None
        self.batch_size = None
        self.early_stop = None
        self.learning_rate_list = None
        self.alpha = None
        self.beta = None
        self.l1 = None
        self.l2 = None
        self.net2net_applied = None
        self.ck_length_saving = None  # Check point
        self.ck_folder = None
        self.dyge_shuffle = None
        self.dyge_resume_training = None
        self.dyge_activation = None

        # sdne_config
        self.sdne_learning_rate = None
        self.sdne_shuffle = None
        self.sdne_load_model = None
        self.sdne_resume_training = None
        self.sdne_activation = None

        #  link_pred_config
        self.show_acc_on_edge = None
        self.top_k = None
        self.drop_node_percent = None

        for key in kwargs:
            self.__setattr__(key, kwargs[key])

    def print(self):
        print(vars(self))
        print("\n== Param ==")
        for k in self.__dict__:
            if type(self.__getattribute__(k)) == int or type(self.__getattribute__(k)) == float or type(
                    self.__getattribute__(k)) == list or type(self.__getattribute__(k)) == bool:
                print(f"\t'{k}': {self.__getattribute__(k)}")
            elif type(self.__getattribute__(k)) == str:
                print(f"\t'{k}': \"{self.__getattribute__(k)}\"")
        print("==========\n")


def _get_activation(act):
    act = str.lower(act)
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Invalid activation name: {act}")


class ModelActivation:
    def __init__(self, hidden_layer_act='leaky_relu', embedding_act='tanh', output_act='relu'):
        self.hidden_layer_act = _get_activation(hidden_layer_act)
        self.embedding_act = _get_activation(embedding_act)
        self.output_act = _get_activation(output_act)
        self.data = {
            'hidden': hidden_layer_act,
            'embedding': embedding_act,
            'output': output_act
        }

    def config(self):
        return self.data


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    sp = {
        'top_k': 10,
        'l1': 0.05
    }
    sp = SettingParam(**sp)
    print(sp.show_acc_on_edge)
    print(sp.l1)
