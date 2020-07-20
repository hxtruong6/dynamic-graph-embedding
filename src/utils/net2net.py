import numpy as np


def net2wider(weights1, bias1, weights2, added_size):
    # random index array whose values from [0, current_size_layer] with size = added_size.
    rand_idx = np.random.randint(weights1.shape[1], size=added_size)
    # count how many index is repeated in rand_idx array (histogram of array.)
    replication_factor = np.bincount(rand_idx)

    new_weights1 = weights1.copy()
    new_bias1 = bias1.copy()
    new_weights2 = weights2.copy()

    for i in range(len(rand_idx)):
        unit_index = rand_idx[i]

        # update wider layer (called layer K)
        new_unit = weights1[:, unit_index]
        new_unit = new_unit[:, np.newaxis]
        new_weights1 = np.concatenate((new_weights1, new_unit), axis=1)
        new_bias1 = np.append(new_bias1, bias1[unit_index])

        # update next wider layer (called layer K+1)
        factor = replication_factor[unit_index] + 1
        new_unit = weights2[unit_index, :] * (1. / factor)
        new_unit = new_unit[np.newaxis, :]
        new_weights2 = np.concatenate((new_weights2, new_unit), axis=0)
        new_weights2[unit_index, :] = new_unit

    return new_weights1, new_bias1, new_weights2


def net2deeper(weights):
    '''

    :param weights: numpy array has shape(inp_size, out_size). input_size and out_size are number of units from source
                    to destination layer
    :return:
    '''
    _, out = weights.shape
    new_weights = np.array(np.eye(out))
    new_bias = np.zeros((out,))
    return new_weights, new_bias
