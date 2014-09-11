import numpy as np

import theano 
import theano.tensor as T

import init


def floatX(arr):
    """
    Shortcut to turn a numpy array into an array with the
    correct dtype for Theano.
    """
    return arr.astype(theano.config.floatX)


def shared_empty(dim=2):
    """
    Shortcut to create an empty Theano shared variable with
    the specified number of dimensions.
    """
    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=theano.config.floatX))


def unique(l):
    """
    Create a new list from l with duplicate entries removed,
    while preserving the original order.
    """
    new_list = []
    for el in l:
        if el not in new_list:
            new_list.append(el)

    return new_list