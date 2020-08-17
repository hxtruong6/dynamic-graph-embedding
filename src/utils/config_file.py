import configparser

from src.utils.setting_param import SettingParam


def handle_int_list(lrs):
    lrs = [float(lr) for lr in lrs.split(',')]
    return lrs


def read_config_file(filepath, config_task, verbose=False):
    '''

    :param filepath:
    :param config_task: Must be provide. One of values: 'stability'|'link_pred'
    :return:
    '''
    if config_task is None or (config_task != "stability" and config_task != "link_pred"):
        raise ValueError("Invalid config_task.")
    config = configparser.ConfigParser()
    config.read(filepath)
    config.sections()

    global_cf = config['global']
    algorithm_cf = config['algorithm']
    train_cf = config['training_config']
    dyge_cf = config['dyngem_config']
    sdne_cf = config['sdne_config']
    link_pred_cf = config['link_pred_config']

    dataset_name = config['global']['dataset_name']

    params = {
        'dataset_name': dataset_name,
        # 'algorithm': {
        'is_dyge': algorithm_cf.getboolean('is_dyge'),
        'is_node2vec': algorithm_cf.getboolean('is_node2vec'),
        'is_sdne': algorithm_cf.getboolean('is_sdne'),

        # 'folder_paths': {
        'dataset_folder': f"./data/{dataset_name}",
        'processed_link_pred_data_folder': f"./saved_data/processed_data/{dataset_name}_{config_task}",

        'dyge_weight_folder': f"./saved_data/models/{dataset_name}_{config_task}",
        'dyge_emb_folder': f"./saved_data/embeddings/{dataset_name}_{config_task}",

        'node2vec_emb_folder': f"./saved_data/node2vec_emb/{dataset_name}_{config_task}",

        'sdne_weight_folder': f"./saved_data/sdne_models/{dataset_name}_{config_task}",
        'sdne_emb_folder': f"./saved_data/sdne_emb/{dataset_name}_{config_task}",

        'global_seed': int(global_cf['global_seed']),

        # Processed data
        'is_load_link_pred_data': global_cf.getboolean('is_load_link_pred_data'),

        # 'training_config': {
        'is_load_dyge_model': train_cf.getboolean('is_load_dyge_model'),
        'specific_dyge_model_index': int(train_cf['specific_dyge_model_index']) if train_cf[
                                                                                       'specific_dyge_model_index'] != "None" else None,
        'dyge_resume_training': train_cf.getboolean('dyge_resume_training'),

        # 'dyge_config': {
        'prop_size': float(dyge_cf['prop_size']),
        'embedding_dim': int(dyge_cf['embedding_dim']),
        'epochs': int(dyge_cf['epochs']),
        'skip_print': int(dyge_cf['skip_print']),
        'batch_size': int(dyge_cf['batch_size']) if dyge_cf['batch_size'] != 'None' else None,
        'early_stop': int(dyge_cf['early_stop']),  # 100
        'learning_rate_list': handle_int_list(dyge_cf['learning_rate_list']),
        'alpha': float(dyge_cf['alpha']),
        'beta': float(dyge_cf['beta']),
        'l1': float(dyge_cf['l1']),
        'l2': float(dyge_cf['l2']),
        'net2net_applied': dyge_cf.getboolean('net2net_applied'),
        'ck_length_saving': int(dyge_cf['ck_length_saving']),
        'ck_folder': f'./saved_data/models/{dataset_name}_{config_task}_ck',
        'dyge_shuffle': dyge_cf.getboolean('dyge_shuffle'),
        'dyge_activation': dyge_cf['dyge_activation'],

        # SDNE
        'sdne_learning_rate': float(sdne_cf['sdne_learning_rate']),
        'sdne_shuffle': sdne_cf.getboolean('sdne_shuffle'),
        'sdne_load_model': sdne_cf.getboolean('sdne_load_model'),
        'sdne_resume_training': sdne_cf.getboolean('sdne_resume_training'),
        'sdne_activation': sdne_cf['sdne_activation'],

        # 'link_pred_config': {
        'show_acc_on_edge': link_pred_cf.getboolean('show_acc_on_edge'),
        'top_k': int(link_pred_cf['top_k']),
        'drop_node_percent': float(link_pred_cf['drop_node_percent']),
    }

    params = SettingParam(**params)
    if verbose:
        params.print()
    return params


if __name__ == '__main__':
    params = read_config_file("../../link_pred_configuration.ini", config_task="link_pred")
    params.print()
