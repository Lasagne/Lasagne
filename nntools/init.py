"""
Functions to create initializers for parameter variables
"""
from numbers import Number

import numpy as np
import theano
import theano.tensor as T

from utils import floatX


class Initializer(object):
    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError


class Normal(Initializer):
    def __init__(self, std=0.01, avg=0.0):
        self.std = std
        self.avg = avg

    def sample(self, shape):
        return floatX(np.random.normal(self.avg, self.std, size=shape))


class Constant(Initializer):
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):   
        return floatX(np.ones(shape) * self.val)


class Sparse(Initializer):
    def __init__(self, sparsity=0.1, std=0.01):
        self.sparsity = sparsity
        self.std = std

    def sample(self, shape):
        if len(shape) != 2:
            raise RuntimeError("sparse initializer only works with shapes of length 2")

        w = floatX(np.zeros(shape))
        n_inputs, n_outputs = shape
        size = int(self.sparsity * n_inputs) # fraction of the number of inputs

        for k in range(n_outputs):
            indices = np.arange(n_inputs)
            np.random.shuffle(indices)
            indices = indices[:size]
            values = floatX(np.random.normal(0.0, self.std, size=size))
            w[indices, k] = values

        return w


class Uniform(Initializer):
    def __init__(self, range=None):
        self.range = range

    def sample(self, shape):
        if self.range is None:
            # no range given, use the Glorot et al. approach
            if len(shape) != 2:
                raise RuntimeError("uniform initializer without parameters only works with shapes of length 2")

            n_inputs, n_outputs = shape
            m = np.sqrt(6.0 / (n_inputs + n_outputs))
            range = (-m, m)

        elif isinstance(self.range, Number):
            range = (-self.range, self.range)

        else:
            range = self.range

        return floatX(np.random.uniform(low=range[0], high=range[1], size=shape))



