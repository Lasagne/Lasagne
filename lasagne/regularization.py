import theano
import theano.tensor as T

import layers


def l2(layer, include_biases=False):
    if include_biases:
        all_params = layers.get_all_params(layer)
    else:
        all_params = layers.get_all_non_bias_params(layer)
    
    return sum(T.sum(p**2) for p in all_params)


# TODO: sparsity regularization