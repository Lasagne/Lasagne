import numpy as np

from .. import utils


__all__ = [
    "get_all_layers",
    "get_all_params",
    "get_all_bias_params",
    "get_all_non_bias_params",
    "count_params",
    "get_all_param_values",
    "set_all_param_values",
]


def get_all_layers(layer):
    """
    This function gathers all layers below one or more given :class:`Layer`
    instances, including the given layer(s). Its main use is to collect all
    layers of a network just given the output layer(s).

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_layers = get_all_layers(l1)
        >>> all_layers == [l1, l_in]
        True
        >>> l2 = DenseLayer(l_in, num_units=10)
        >>> all_layers = get_all_layers([l2, l1])
        >>> all_layers == [l2, l1, l_in]
        True

    :parameters:
        - layer : Layer
            the :class:`Layer` instance for which to gather all layers feeding
            into it, or a list of :class:`Layer` instances.

    :returns:
        - layers : list
            a list of :class:`Layer` instances feeding into the given
            instance(s) either directly or indirectly, and the given
            instance(s) themselves.
    """
    if isinstance(layer, (list, tuple)):
        layers = list(layer)
    else:
        layers = [layer]
    layers_to_expand = list(layers)
    while len(layers_to_expand) > 0:
        current_layer = layers_to_expand.pop(0)
        children = []

        if hasattr(current_layer, 'input_layers'):
            children = current_layer.input_layers
        elif hasattr(current_layer, 'input_layer'):
            children = [current_layer.input_layer]

        # filter the layers that have already been visited, and remove None
        # elements (for layers without incoming layers)
        children = [child for child in children
                    if child not in layers and
                    child is not None]
        layers_to_expand.extend(children)
        layers.extend(children)

    return layers


def get_all_params(layer):
    """
    This function gathers all learnable parameters of all layers below one
    or more given :class:`Layer` instances, including the layer(s) itself.
    Its main use is to collect all parameters of a network just given the
    output layer(s).

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_params = get_all_params(l1)
        >>> all_params == [l1.W, l1.b]
        True

    :parameters:
        - layer : Layer
            the :class:`Layer` instance for which to gather all parameters,
            or a list of :class:`Layer` instances.

    :returns:
        - params : list
            a list of Theano shared variables representing the parameters.
    """
    layers = get_all_layers(layer)
    params = sum([l.get_params() for l in layers], [])
    return utils.unique(params)


def get_all_bias_params(layer):
    """
    This function gathers all learnable bias parameters of all layers below one
    or more given :class:`Layer` instances, including the layer(s) itself.

    This is useful for situations where the biases should be treated
    separately from other parameters, e.g. they are typically excluded from
    L2 regularization.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_params = get_all_bias_params(l1)
        >>> all_params == [l1.b]
        True

    :parameters:
        - layer : Layer
            the :class:`Layer` instance for which to gather all bias
            parameters, or a list of :class:`Layer` instances.

    :returns:
        - params : list
            a list of Theano shared variables representing the bias parameters.

    """
    layers = get_all_layers(layer)
    params = sum([l.get_bias_params() for l in layers], [])
    return utils.unique(params)


def get_all_non_bias_params(layer):
    """
    This function gathers all learnable non-bias parameters of all layers below
    one or more given :class:`Layer` instances, including the layer(s) itself.

    This is useful for situations where the biases should be treated
    separately from other parameters, e.g. they are typically excluded from
    L2 regularization.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_params = get_all_non_bias_params(l1)
        >>> all_params == [l1.W]
        True

    :parameters:
        - layer : Layer
            the :class:`Layer` instance for which to gather all non-bias
            parameters, or a list of :class:`Layer` instances.

    :returns:
        - params : list
            a list of Theano shared variables representing the non-bias
            parameters.

    """
    all_params = get_all_params(layer)
    all_bias_params = get_all_bias_params(layer)
    return [p for p in all_params if p not in all_bias_params]


def count_params(layer):
    """
    This function counts all learnable parameters (i.e. the number of scalar
    values) of all layers below one or more given :class:`Layer` instances,
    including the layer(s) itself.

    This is useful to compare the capacity of various network architectures.
    All parameters returned by the :class:`Layer`s' `get_params` methods are
    counted, including biases.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> param_count = count_params(l1)
        >>> param_count
        1050
        >>> param_count == 20 * 50 + 50  # 20 input * 50 units + 50 biases
        True

    :parameters:
        - layer : Layer
            the :class:`Layer` instance for which to count the parameters,
            or a list of :class:`Layer` instances.
    :returns:
        - count : int
            the total number of learnable parameters.

    """
    params = get_all_params(layer)
    shapes = [p.get_value().shape for p in params]
    counts = [np.prod(shape) for shape in shapes]
    return sum(counts)


def get_all_param_values(layer):
    """
    This function returns the values of the parameters of all layers below one
    or more given :class:`Layer` instances, including the layer(s) itself.

    This function can be used in conjunction with set_all_param_values to save
    and restore model parameters.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_param_values = get_all_param_values(l1)
        >>> (all_param_values[0] == l1.W.get_value()).all()
        True
        >>> (all_param_values[1] == l1.b.get_value()).all()
        True

    :parameters:
        - layer : Layer
            the :class:`Layer` instance for which to gather all parameter
            values, or a list of :class:`Layer` instances.

    :returns:
        - param_values : list of numpy.array
            a list of numpy arrays representing the parameter values.
    """
    params = get_all_params(layer)
    return [p.get_value() for p in params]


def set_all_param_values(layer, values):
    """
    Given a list of numpy arrays, this function sets the parameters of all
    layers below one or more given :class:`Layer` instances (including the
    layer(s) itself) to the given values.

    This function can be used in conjunction with get_all_param_values to save
    and restore model parameters.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_param_values = get_all_param_values(l1)
        >>> # all_param_values is now [l1.W.get_value(), l1.b.get_value()]
        >>> # ...
        >>> set_all_param_values(l1, all_param_values)
        >>> # the parameter values are restored.

    :parameters:
        - layer : Layer
            the :class:`Layer` instance for which to set all parameter
            values, or a list of :class:`Layer` instances.
        - values : list of numpy.array
    """
    params = get_all_params(layer)
    for p, v in zip(params, values):
        p.set_value(v)
