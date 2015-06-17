"""
Provides some minimal help with building loss expressions for training or
validating a neural network.

Three functions build element- or item-wise loss expressions from network
predictions and targets:

.. autosummary::
    :nosignatures:

    binary_crossentropy
    categorical_crossentropy
    squared_error

A convenience function aggregates such losses into a scalar expression
suitable for differentiation:

.. autosummary::
    :nosignatures:

    aggregate

Note that these functions only serve to write more readable code, but are
completely optional. Essentially, any differentiable scalar Theano expression
can be used as a training objective.

Examples
--------
Assuming you have a simple neural network for 3-way classification:

>>> from lasagne.layers import InputLayer, DenseLayer, get_output
>>> from lasagne.nonlinearities import softmax, rectify
>>> l_in = InputLayer((100, 20))
>>> l_hid = DenseLayer(l_in, num_units=30, nonlinearity=rectify)
>>> l_out = DenseLayer(l_hid, num_units=3, nonlinearity=softmax)

And Theano variables representing your network input and targets:

>>> import theano
>>> data = theano.tensor.matrix('data')
>>> targets = theano.tensor.matrix('targets')

You'd first construct an element-wise loss expression:

>>> from lasagne.objectives import categorical_crossentropy, aggregate
>>> predictions = get_output(l_out, data)
>>> loss = categorical_crossentropy(predictions, targets)

Then aggregate it into a scalar (you could also just call ``mean()`` on it):

>>> loss = aggregate(loss, mode='mean')

Finally, this gives a loss expression you can pass to any of the update
methods in :mod:`lasagne.updates`. For validation of a network, you will
usually want to repeat these steps with deterministic network output, i.e.,
without dropout or any other nondeterministic computation in between:

>>> test_predictions = get_output(l_out, data, deterministic=True)
>>> test_loss = categorical_crossentropy(test_predictions, targets)
>>> test_loss = aggregate(test_loss)

This gives a loss expression good for monitoring validation error.
"""

import theano.tensor.nnet

from lasagne.layers import get_output

__all__ = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "squared_error",
    "aggregate",
    "mse", "Objective", "MaskedObjective",  # deprecated
]


def binary_crossentropy(predictions, targets):
    """Computes the binary cross-entropy between predictions and targets.

    .. math:: L = -t \\log(p) - (1 - t) \\log(1 - p)

    Parameters
    ----------
    predictions : Theano tensor
        Predictions in (0, 1), such as sigmoidal output of a neural network.
    targets : Theano tensor
        Targets in [0, 1], such as ground truth labels.

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary cross-entropy.

    Notes
    -----
    This is the loss function of choice for binary classification problems
    and sigmoid output units.
    """
    return theano.tensor.nnet.binary_crossentropy(predictions, targets)


def categorical_crossentropy(predictions, targets):
    """Computes the categorical cross-entropy between predictions and targets.

    .. math:: L_i = - \\sum_j{t_{i,j} \\log(p_{i,j})}

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a vector of int giving the correct class index per data point.

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical cross-entropy.

    Notes
    -----
    This is the loss function of choice for multi-class classification
    problems and softmax output units. For hard targets, i.e., targets
    that assign all of the probability to a single class per data point,
    providing a vector of int for the targets is usually slightly more
    efficient than providing a matrix with a single 1.0 per row.
    """
    return theano.tensor.nnet.categorical_crossentropy(predictions, targets)


def squared_error(a, b):
    """Computes the element-wise squared difference between two tensors.

    .. math:: L = (p - t)^2

    Parameters
    ----------
    a, b : Theano tensor
        The tensors to compute the squared difference between.

    Returns
    -------
    Theano tensor
        An expression for the item-wise squared difference.

    Notes
    -----
    This is the loss function of choice for many regression problems
    or auto-encoders with linear output units.
    """
    return (a - b)**2


def aggregate(loss, weights=None, mode='mean'):
    """Aggregates an element- or item-wise loss to a scalar loss.

    Parameters
    ----------
    loss : Theano tensor
        The loss expression to aggregate.
    weights : Theano tensor, optional
        The weights for each element or item, must be broadcastable to
        the same shape as `loss` if given. If omitted, all elements will
        be weighted the same.
    mode : {'mean', 'sum', 'normalized_sum'}
        Whether to aggregate by averaging, by summing or by summing and
        dividing by the total weights (which requires `weights` to be given).

    Returns
    -------
    Theano scalar
        A scalar loss expression suitable for differentiation.

    Notes
    -----
    By supplying binary weights (i.e., only using values 0 and 1), this
    function can also be used for masking out particular entries in the
    loss expression. Note that masked entries still need to be valid
    values, not-a-numbers (NaNs) will propagate through.

    When applied to batch-wise loss expressions, setting `mode` to
    ``'normalized_sum'`` ensures that the loss per batch is of a similar
    magnitude, independent of associated weights. However, it means that
    a given datapoint contributes more to the loss when it shares a batch
    with low-weighted or masked datapoints than with high-weighted ones.
    """
    if weights is not None:
        loss = loss * weights
    if mode == 'mean':
        return loss.mean()
    elif mode == 'sum':
        return loss.sum()
    elif mode == 'normalized_sum':
        if weights is None:
            raise ValueError("require weights for mode='normalized_sum'")
        return loss.sum() / weights.sum()
    else:
        raise ValueError("mode must be 'mean', 'sum' or 'normalized_sum', "
                         "got %r" % mode)


def mse(x, t):  # pragma no cover
    """Deprecated. Use :func:`squared_error()` instead."""
    import warnings
    warnings.warn("lasagne.objectives.mse() is deprecated and will be removed "
                  "for the first release of Lasagne. Use "
                  "lasagne.objectives.squared_error() instead.", stacklevel=2)
    return squared_error(x, t)


class Objective(object):  # pragma no cover
    """
    Deprecated. See docstring of :mod:`lasagne.objectives` for alternatives.
    """

    def __init__(self, input_layer, loss_function=squared_error,
                 aggregation='mean'):
        import warnings
        warnings.warn("lasagne.objectives.Objective is deprecated and "
                      "will be removed for the first release of Lasagne. For "
                      "alternatives, please see: "
                      "http://lasagne.readthedocs.org/en/latest/"
                      "modules/objectives.html", stacklevel=2)
        import theano.tensor as T
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        self.aggregation = aggregation

    def get_loss(self, input=None, target=None, aggregation=None, **kwargs):
        from lasagne.layers import get_output
        network_output = get_output(self.input_layer, input, **kwargs)

        if target is None:
            target = self.target_var
        if aggregation is None:
            aggregation = self.aggregation

        losses = self.loss_function(network_output, target)

        return aggregate(losses, mode=aggregation)


class MaskedObjective(object):  # pragma no cover
    """
    Deprecated. See docstring of :mod:`lasagne.objectives` for alternatives.
    """

    def __init__(self, input_layer, loss_function=mse, aggregation='mean'):
        import warnings
        warnings.warn("lasagne.objectives.MaskedObjective is deprecated and "
                      "will be removed for the first release of Lasagne. For "
                      "alternatives, please see: "
                      "http://lasagne.readthedocs.org/en/latest/"
                      "modules/objectives.html", stacklevel=2)
        import theano.tensor as T
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        self.mask_var = T.matrix("mask")
        self.aggregation = aggregation

    def get_loss(self, input=None, target=None, mask=None,
                 aggregation=None, **kwargs):
        from lasagne.layers import get_output
        network_output = get_output(self.input_layer, input, **kwargs)

        if target is None:
            target = self.target_var
        if mask is None:
            mask = self.mask_var
        if aggregation is None:
            aggregation = self.aggregation

        losses = self.loss_function(network_output, target)

        return aggregate(losses, mask, mode=aggregation)
