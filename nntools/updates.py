"""
Functions to generate Theano update dictionaries for training.
"""

import numpy as np

import theano
import theano.tensor as T


def sgd(loss, all_params, learning_rate):
    all_grads = [theano.grad(loss, param) for param in all_params]
    updates = []
    
    for param_i, grad_i in zip(all_params, all_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    return updates


def momentum(loss, all_params, learning_rate, momentum=0.9, weight_decay=0.0):
    all_grads = [theano.grad(loss, param) for param in all_params]
    updates = []
    
    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - weight_decay * learning_rate * param_i  - learning_rate * grad_i
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))

    return updates


# using the alternative formulation of nesterov momentum described at https://github.com/lisa-lab/pylearn2/pull/136
# such that the gradient can be evaluated at the current parameters.
def nesterov_momentum(loss, all_params, learning_rate, momentum=0.9, weight_decay=0.0):
    all_grads = [theano.grad(loss, param) for param in all_params]
    updates = []
    
    for param_i, grad_i in zip(all_params, all_grads):
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))

    return updates


def adagrad(loss, all_params, learning_rate=1.0, epsilon=1e-6):
    """
    epsilon is not included in the typical formula, 
    See "Notes on AdaGrad" by Chris Dyer for more info.
    """
    all_grads = [theano.grad(loss, param) for param in all_params]
    all_accumulators = [theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in all_params]

    updates = []
    for param_i, grad_i, acc_i in zip(all_params, all_grads, all_accumulators):
        acc_i_new = acc_i + grad_i**2
        updates.append((acc_i, acc_i_new))
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc_i_new + epsilon)))

    return updates


def rmsprop(loss, all_params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    epsilon is not included in the description in Hinton's video,
    but to prevent problems with relus repeatedly having 0 gradients, it is included here.

    Watch this video for more info: http://www.youtube.com/watch?v=O3sxAc4hxZU (formula at 5:20)
    also check http://climin.readthedocs.org/en/latest/rmsprop.html
    """
    all_grads = [theano.grad(loss, param) for param in all_params]
    all_accumulators = [theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in all_params]

    updates = []
    for param_i, grad_i, acc_i in zip(all_params, all_grads, all_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i**2
        updates.append((acc_i, acc_i_new))
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc_i_new + epsilon)))

    return updates


def adadelta(loss, all_params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    """
    in the paper, no learning rate is considered (so learning_rate=1.0). Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does not become 0).

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to work for multiple datasets (MNIST, speech).

    see "Adadelta: an adaptive learning rate method" by Matthew Zeiler for more info.
    """
    all_grads = [theano.grad(loss, param) for param in all_params]
    all_accumulators = [theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in all_params]
    all_delta_accumulators = [theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)) for param in all_params]

    # all_accumulators: accumulate gradient magnitudes
    # all_delta_accumulators: accumulate update magnitudes (recursive!)

    updates = []
    for param_i, grad_i, acc_i, acc_delta_i in zip(all_params, all_grads, all_accumulators, all_delta_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i**2
        updates.append((acc_i, acc_i_new))

        update_i = grad_i * T.sqrt(acc_delta_i + epsilon) / T.sqrt(acc_i_new + epsilon) # use the 'old' acc_delta here
        updates.append((param_i, param_i - learning_rate * update_i))

        acc_delta_i_new = rho * acc_delta_i + (1 - rho) * update_i**2
        updates.append((acc_delta_i, acc_delta_i_new))

    return updates

