"""
Functions to generate Theano update dictionaries for training.
"""

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T


__all__ = [
    "sgd",
    "apply_momentum",
    "momentum",
    "apply_nesterov_momentum",
    "nesterov_momentum",
    "adagrad",
    "rmsprop",
    "adadelta",
    "norm_constraint",
]


def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for

    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.
    """
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)


def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates.

    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates


def apply_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary that includes momentum terms.

    Generates update expressions of the form:
    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + velocity``

    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions
    params : iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    See Also
    --------
    momentum : Shortcut applying momentum to SGD updates
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates


def momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum.

    Generates update expressions of the form:
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + velocity``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
         A dictionary mapping each parameter to its update expression

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    See Also
    --------
    apply_momentum : Generic function applying momentum to updates
    nesterov_momentum : Nesterov's variant of SGD with momentum
    """
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_momentum(updates, momentum=momentum)


def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    """Returns a modified update dictionary including Nesterov momentum.

    Generates update expressions of the form:
    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + momentum * velocity + updates[param] - param``

    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions
    params : iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.

    See Also
    --------
    nesterov_momentum : Shortcut applying Nesterov momentum to SGD updates
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return updates


def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum.

    Generates update expressions of the form:
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Returns
    -------
    OrderedDict
         A dictionary mapping each parameter to its update expression

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.

    See Also
    --------
    apply_nesterov_momentum : Function applying momentum to updates
    """
    updates = sgd(loss_or_grads, params, learning_rate)
    return apply_nesterov_momentum(updates, momentum=momentum)


def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    """
    epsilon is not included in the typical formula,
    See "Notes on AdaGrad" by Chris Dyer for more info.
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates


def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    epsilon is not included in the description in Hinton's video,
    but to prevent problems with relus repeatedly having 0 gradients,
    it is included here.

    Watch this video for more info: http://www.youtube.com/watch?v=O3sxAc4hxZU
    (formula at 5:20)
    also check http://climin.readthedocs.org/en/latest/rmsprop.html
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates


def adadelta(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
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
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    """Max weight norm constraints and gradient clipping

    This takes a TensorVariable and rescales it so that incoming weight
    norms are below a specified constraint value. Vectors violating the
    constraint are rescaled so that they are within the allowed range.

    Parameters
    ----------
    tensor_var : TensorVariable
        Theano expression for update, gradient, or other quantity.
    max_norm : scalar
        This value sets the maximum allowed value of any norm in
        `tensor_var`.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `tensor_var`. When this is not specified and `tensor_var` is a
        matrix (2D), this is set to `(0,)`. If `tensor_var` is a 3D, 4D or
        5D tensor, it is set to a tuple listing all axes but axis 0. The
        former default is useful for working with dense layers, the latter
        is useful for 1D, 2D and 3D convolutional layers.
        (Optional)
    epsilon : scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.

    Returns
    -------
    TensorVariable
        Input `tensor_var` with rescaling applied to weight vectors
        that violate the specified constraints.

    Examples
    --------
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

    Notes
    -----
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
