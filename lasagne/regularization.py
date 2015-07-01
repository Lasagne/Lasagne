"""
Functions to apply regularization to the weights in a network.

We provide functions to calculate the L1 and L2 penalty. Penalty functions
take a tensor as input and calculate the penalty contribution from that tensor:

.. autosummary::
    :nosignatures:

    l1
    l2

A helper function can be used to apply a penalty function to a tensor or a
list of tensors:

.. autosummary::
    :nosignatures:

    apply_penalty

Finally we provide two helper functions for applying a penalty function to the
parameters in a layer or the parameters in a group of layers:

.. autosummary::
    :nosignatures:

    regularize_layer_params_weighted
    regularize_network_params

Examples
--------
>>> import lasagne
>>> import theano.tensor as T
>>> import theano
>>> from lasagne.nonlinearities import softmax
>>> from lasagne.layers import InputLayer, DenseLayer, get_output
>>> from lasagne.regularization import regularize_layer_params_weighted, l2, l1
>>> from lasagne.regularization import regularize_layer_params
>>> layer_in = InputLayer((100, 20))
>>> layer1 = DenseLayer(layer_in, num_units=3)
>>> layer2 = DenseLayer(layer1, num_units=5, nonlinearity=softmax)
>>> x = T.matrix('x')  # shp: num_batch x num_features
>>> y = T.ivector('y') # shp: num_batch
>>> l_out = get_output(layer2, x)
>>> loss = T.mean(T.nnet.categorical_crossentropy(l_out, y))
>>> layers = {layer1: 0.1, layer2: 0.5}
>>> l2_penalty = regularize_layer_params_weighted(layers, l2)
>>> l1_penalty = regularize_layer_params(layer2, l1) * 1e-4
>>> loss = loss + l2_penalty + l1_penalty
"""
import theano.tensor as T
from .layers import Layer, get_all_params


def l1(x):
    """Computes the L1 norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        l1 norm (sum of absolute values of elements)
    """
    return T.sum(abs(x))


def l2(x):
    """Computes the squared L2 norm of a tensor

    Parameters
    ----------
    x : Theano tensor

    Returns
    -------
    Theano scalar
        squared l2 norm (sum of squared values of elements)
    """
    return T.sum(x**2)


def apply_penalty(tensor_or_tensors, penalty, **kwargs):
    """
    Computes the total cost for applying a specified penalty
    to a tensor or group of tensors.

    Parameters
    ----------
    tensor_or_tensors : Theano tensor or list of tensors
    penalty : callable
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the total penalty cost
    """
    try:
        return sum(penalty(x, **kwargs) for x in tensor_or_tensors)
    except (TypeError, ValueError):
        return penalty(tensor_or_tensors, **kwargs)


def regularize_layer_params(layer, penalty,
                            tags={'regularizable': True}, **kwargs):
    """
    Computes a regularization cost by applying a penalty to the parameters
    of a layer or group of layers.

    Parameters
    ----------
    layer : a :class:`Layer` instances or list of layers.
    penalty : callable
    tags: dict
        Tag specifications which filter the parameters of the layer or layers.
        By default, only parameters with the `regularizable` tag are included.
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the cost
    """
    layers = [layer, ] if isinstance(layer, Layer) else layer
    all_params = []

    for layer in layers:
        all_params += layer.get_params(**tags)

    return apply_penalty(all_params, penalty, **kwargs)


def regularize_layer_params_weighted(layers, penalty,
                                     tags={'regularizable': True}, **kwargs):
    """
    Computes a regularization cost by applying a penalty to the parameters
    of a layer or group of layers, weighted by a coefficient for each layer.

    Parameters
    ----------
    layers : dict
        A mapping from :class:`Layer` instances to coefficients.
    penalty : callable
    tags: dict
        Tag specifications which filter the parameters of the layer or layers.
        By default, only parameters with the `regularizable` tag are included.
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the cost
    """
    return sum(coeff * apply_penalty(layer.get_params(**tags),
                                     penalty,
                                     **kwargs)
               for layer, coeff in layers.items()
               )


def regularize_network_params(layer, penalty,
                              tags={'regularizable': True}, **kwargs):
    """
    Computes a regularization cost by applying a penalty to the parameters
    of all layers in a network.

    Parameters
    ----------
    layer : a :class:`Layer` instance.
        Parameters of this layer and all layers below it will be penalized.
    penalty : callable
    tags: dict
        Tag specifications which filter the parameters of the layer or layers.
        By default, only parameters with the `regularizable` tag are included.
    **kwargs
        keyword arguments passed to penalty.

    Returns
    -------
    Theano scalar
        a scalar expression for the cost
    """
    return apply_penalty(get_all_params(layer, **tags), penalty, **kwargs)
