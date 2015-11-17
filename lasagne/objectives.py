"""
Provides some minimal help with building loss expressions for training or
validating a neural network.

Five functions build element- or item-wise loss expressions from network
predictions and targets:

.. autosummary::
    :nosignatures:

    binary_crossentropy
    categorical_crossentropy
    squared_error
    binary_hinge_loss
    multiclass_hinge_loss

A convenience function aggregates such losses into a scalar expression
suitable for differentiation:

.. autosummary::
    :nosignatures:

    aggregate

Note that these functions only serve to write more readable code, but are
completely optional. Essentially, any differentiable scalar Theano expression
can be used as a training objective.

Finally, two functions compute evaluation measures that are useful for
validation and testing only, not for training:

.. autosummary::
   :nosignatures:

   binary_accuracy
   categorical_accuracy

Those can also be aggregated into a scalar expression if needed.

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
    "binary_hinge_loss",
    "multiclass_hinge_loss",
    "binary_accuracy",
    "categorical_accuracy"
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
    a given data point contributes more to the loss when it shares a batch
    with low-weighted or masked data points than with high-weighted ones.
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


def binary_hinge_loss(predictions, targets, binary=True, delta=1):
    """Computes the binary hinge loss between predictions and targets.

    .. math:: L_i = \\max(0, \\delta - t_i p_i)

    Parameters
    ----------
    predictions : Theano tensor
        Predictions in (0, 1), such as sigmoidal output of a neural network.
    targets : Theano tensor
        Targets in {0, 1} (or in {-1, 1} depending on `binary`), such as
        ground truth labels.
    binary : bool, default True
        ``True`` if targets are in {0, 1}, ``False`` if they are in {-1, 1}
    delta : scalar, default 1
        The hinge loss margin

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary hinge loss

    Notes
    -----
    This is an alternative to the binary cross-entropy loss for binary
    classification problems
    """
    if binary:
        targets = 2 * targets - 1
    return theano.tensor.nnet.relu(delta - predictions * targets)


def multiclass_hinge_loss(predictions, targets, delta=1):
    """Computes the multi-class hinge loss between predictions and targets.

    .. math:: L_i = \\max_{j \\not = p_i} (0, t_j - t_{p_i} + \\delta)

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of one-hot encoding of the correct class in the same
        layout as predictions (non-binary targets in [0, 1] do not work!)
    delta : scalar, default 1
        The hinge loss margin

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise multi-class hinge loss

    Notes
    -----
    This is an alternative to the categorical cross-entropy loss for
    multi-class classification problems
    """
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    corrects = predictions[targets.nonzero()]
    rest = theano.tensor.reshape(predictions[(1-targets).nonzero()],
                                 (-1, num_cls-1))
    rest = theano.tensor.max(rest, axis=1)
    return theano.tensor.nnet.relu(rest - corrects + delta)


def binary_accuracy(predictions, targets, threshold=0.5):
    """Computes the binary accuracy between predictions and targets.

    .. math:: L_i = \\mathbb{I}(t_i = \mathbb{I}(p_i \\ge \\alpha))

    Parameters
    ----------
    predictions : Theano tensor
        Predictions in [0, 1], such as a sigmoidal output of a neural network,
        giving the probability of the positive class
    targets : Theano tensor
        Targets in {0, 1}, such as ground truth labels.
    threshold : scalar, default: 0.5
        Specifies at what threshold to consider the predictions being of the
        positive class

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary accuracy in {0, 1}

    Notes
    -----
    This objective function should not be used with a gradient calculation;
    its gradient is zero everywhere. It is intended as a convenience for
    validation and testing, not training.

    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    predictions = theano.tensor.ge(predictions, threshold)
    return theano.tensor.eq(predictions, targets)


def categorical_accuracy(predictions, targets, top_k=1):
    """Computes the categorical accuracy between predictions and targets.

    .. math:: L_i = \\mathbb{I}(t_i = \\operatorname{argmax}_c p_{i,c})

    Can be relaxed to allow matches among the top :math:`k` predictions:

    .. math::
        L_i = \\mathbb{I}(t_i \\in \\operatorname{argsort}_c (-p_{i,c})_{:k})

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of 1 hot encoding of the correct class in the same
        layout as predictions
    top_k : int
        Regard a prediction to be correct if the target class is among the
        `top_k` largest class probabilities. For the default value of 1, a
        prediction is correct only if the target class is the most probable.

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical accuracy in {0, 1}

    Notes
    -----
    This is a strictly non differential function as it includes an argmax.
    This objective function should never be used with a gradient calculation.
    It is intended as a convenience for validation and testing not training.

    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    if targets.ndim == predictions.ndim:
        targets = theano.tensor.argmax(targets, axis=-1)
    elif targets.ndim != predictions.ndim - 1:
        raise TypeError('rank mismatch between targets and predictions')

    if top_k == 1:
        # standard categorical accuracy
        top = theano.tensor.argmax(predictions, axis=-1)
        return theano.tensor.eq(top, targets)
    else:
        # top-k accuracy
        top = theano.tensor.argsort(predictions, axis=-1)
        # (Theano cannot index with [..., -top_k:], we need to simulate that)
        top = top[[slice(None) for _ in range(top.ndim - 1)] +
                  [slice(-top_k, None)]]
        targets = theano.tensor.shape_padaxis(targets, axis=-1)
        return theano.tensor.any(theano.tensor.eq(top, targets), axis=-1)
