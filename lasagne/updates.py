"""
Functions to generate Theano update dictionaries for training.
"""

import numpy as np

import theano
import theano.tensor as T


def sgd(loss, all_params, learning_rate):
    all_grads = theano.grad(loss, all_params)
    updates = []

    for param_i, grad_i in zip(all_params, all_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    return updates


def momentum(loss, all_params, learning_rate, momentum=0.9):
    all_grads = theano.grad(loss, all_params)
    updates = []

    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape,
                                          dtype=theano.config.floatX),
                                 broadcastable=param_i.broadcastable)
        v = momentum * mparam_i - learning_rate * grad_i
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))

    return updates


# using the alternative formulation of nesterov momentum described at
# https://github.com/lisa-lab/pylearn2/pull/136
# such that the gradient can be evaluated at the current parameters.
def nesterov_momentum(loss, all_params, learning_rate, momentum=0.9):
    all_grads = theano.grad(loss, all_params)
    updates = []

    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape,
                                          dtype=theano.config.floatX),
                                 broadcastable=param_i.broadcastable)
        v = momentum * mparam_i - learning_rate * grad_i  # new momemtum
        w = param_i + momentum * v - learning_rate * grad_i  # new param values
        updates.append((mparam_i, v))
        updates.append((param_i, w))

    return updates


def adagrad(loss, all_params, learning_rate=1.0, epsilon=1e-6):
    """
    epsilon is not included in the typical formula,
    See "Notes on AdaGrad" by Chris Dyer for more info.
    """
    all_grads = theano.grad(loss, all_params)
    all_accumulators = [theano.shared(np.zeros(param.get_value().shape,
                                               dtype=theano.config.floatX),
                                      broadcastable=param.broadcastable)
                        for param in all_params]

    updates = []
    for param_i, grad_i, acc_i in zip(all_params, all_grads, all_accumulators):
        acc_i_new = acc_i + grad_i ** 2
        updates.append((acc_i, acc_i_new))
        updates.append((param_i, (param_i - learning_rate * grad_i /
                                  T.sqrt(acc_i_new + epsilon))))

    return updates


def rmsprop(loss, all_params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    epsilon is not included in the description in Hinton's video,
    but to prevent problems with relus repeatedly having 0 gradients,
    it is included here.

    Watch this video for more info: http://www.youtube.com/watch?v=O3sxAc4hxZU
    (formula at 5:20)
    also check http://climin.readthedocs.org/en/latest/rmsprop.html
    """
    all_grads = theano.grad(loss, all_params)
    all_accumulators = [theano.shared(np.zeros(param.get_value().shape,
                                               dtype=theano.config.floatX),
                                      broadcastable=param.broadcastable)
                        for param in all_params]

    updates = []
    for param_i, grad_i, acc_i in zip(all_params, all_grads, all_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
        updates.append((acc_i, acc_i_new))
        updates.append((param_i, (param_i - learning_rate * grad_i /
                                  T.sqrt(acc_i_new + epsilon))))

    return updates


def adadelta(loss, all_params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    """
    in the paper, no learning rate is considered (so learning_rate=1.0).
    Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does
    not become 0).

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
    work for multiple datasets (MNIST, speech).

    see "Adadelta: an adaptive learning rate method" by Matthew Zeiler
    for more info.
    """
    all_grads = theano.grad(loss, all_params)
    all_accumulators = [theano.shared(np.zeros(param.get_value().shape,
                                               dtype=theano.config.floatX),
                                      broadcastable=param.broadcastable)
                        for param in all_params]

    all_delta_accumulators = [
        theano.shared(np.zeros(param.get_value().shape,
                               dtype=theano.config.floatX),
                      broadcastable=param.broadcastable)
        for param in all_params
    ]

    # all_accumulators: accumulate gradient magnitudes
    # all_delta_accumulators: accumulate update magnitudes (recursive!)

    updates = []
    for param_i, grad_i, acc_i, acc_delta_i in zip(all_params,
                                                   all_grads,
                                                   all_accumulators,
                                                   all_delta_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
        updates.append((acc_i, acc_i_new))

        update_i = (grad_i * T.sqrt(acc_delta_i + epsilon) /
                    # use the 'old' acc_delta here
                    T.sqrt(acc_i_new + epsilon))
        updates.append((param_i, param_i - learning_rate * update_i))

        acc_delta_i_new = rho * acc_delta_i + (1 - rho) * update_i ** 2
        updates.append((acc_delta_i, acc_delta_i_new))

    return updates


def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    """
    Max weight norm constraints and gradient clipping

    This takes a TensorVariable and rescales it so that incoming weight
    norms are below a specified constraint value. Vectors violating the
    constraint are rescaled so that they are within the allowed range.

    :parameters:
        - tensor_var : TensorVariable
            Theano expression for update, gradient, or other quantity.
        - max_norm : scalar
            This value sets the maximum allowed value of any norm in
            `tensor_var`.
        - norm_axes : sequence (list or tuple)
            The axes over which to compute the norm.  This overrides the
            default norm axes defined for the number of dimensions
            in `tensor_var`. When this is not specified and `tensor_var` is a
            matrix (2D), this is set to `(0,)`. If `tensor_var` is a 3D, 4D or
            5D tensor, it is set to a tuple listing all axes but axis 0. The
            former default is useful for working with dense layers, the latter
            is useful for 1D, 2D and 3D convolutional layers.
            (Optional)
        - epsilon : scalar
            Value used to prevent numerical instability when dividing by
            very small or zero norms.
            (Optional)

    :returns:
        - constrained_output : TensorVariable
            Input `tensor_var` with rescaling applied to weight vectors
            that violate the specified constraints.

    :usage:
        >>> param = theano.shared(
        ...     np.random.randn(100, 200).astype(theano.config.floatX))
        >>> update = param + 100
        >>> update = norm_constraint(update, 10)
        >>> func = theano.function([], [], updates=[(param, update)])
        >>> # Apply constrained update
        >>> _ = func()
        >>> from lasagne.utils import compute_norms
        >>> norms = compute_norms(param.get_value())
        >>> np.isclose(np.max(norms), 10)
        True

    :note:
        When `norm_axes` is not specified, the axes over which the norm is
        computed depend on the dimensionality of the input variable. If it is
        2D, it is assumed to come from a dense layer, and the norm is computed
        over axis 0. If it is 3D, 4D or 5D, it is assumed to come from a
        convolutional layer and the norm is computed over all trailing axes
        beyond axis 0. For other uses, you should explicitly specify the axes
        over which to compute the norm using `norm_axes`.
    """
    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(ndim)
        )

    dtype = np.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = T.clip(norms, 0, dtype(max_norm))
    constrained_output = \
        (tensor_var * (target_norms / (dtype(epsilon) + norms)))

    return constrained_output
