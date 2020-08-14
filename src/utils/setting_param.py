class SettingParam(object):

    def __init__(self, **kwargs):
        self.global_seed = None

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

        #  training_config
        self.is_load_link_pred_data = None
        self.is_load_dyge_model = None
        self.is_load_n2v_model = None
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

        # sdne_config
        self.sdne_learning_rate = None
        self.sdne_shuffle = None
        self.sdne_load_model = None
        self.sdne_resume_training = None

        #  link_pred_config
        self.show_acc_on_edge = None
        self.top_k = None
        self.drop_node_percent = None

        for key in kwargs:
            self.__setattr__(key, kwargs[key])


if __name__ == '__main__':
    sp = {
        'top_k': 10,
        'l1': 0.05
    }
    sp = SettingParam(**sp)
    print(sp.show_acc_on_edge)
    print(sp.l1)
