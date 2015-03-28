"""
Functions to create initializers for parameter variables
"""

from numbers import Number

import numpy as np

from .utils import floatX


class Initializer(object):
    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()


class Normal(Initializer):
    def __init__(self, std=0.01, avg=0.0):
        self.std = std
        self.avg = avg

    def sample(self, shape):
        return floatX(np.random.normal(self.avg, self.std, size=shape))


class GlorotNormal(Normal):
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

    def sample(self, shape):
        # This code makes some assumptions about the meanings of
        # the different dimensions, which hold for
        # layers.DenseLayer and layers.Conv*DLayer, but not
        # necessarily for other layer types.
        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])
        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return floatX(np.random.normal(0.0, std, size=shape))


class GlorotNormal_c01b(GlorotNormal):
    def sample(self, shape):
        if len(shape) != 4:
            raise RuntimeError(
                "This initializer only works with shapes of length 4")

        n1, n2 = shape[0], shape[3]
        receptive_field_size = shape[1] * shape[2]
        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return floatX(np.random.normal(0.0, std, size=shape))


class Uniform(Initializer):
    def __init__(self, range=0.01):
        import warnings
        warnings.warn("The uniform initializer no longer uses Glorot et al.'s "
                      "approach to determine the bounds, but defaults to the "
                      "range (-0.01, 0.01) instead. Please use the new "
                      "GlorotUniform initializer to get the old behavior. "
                      "GlorotUniform is now the default for all layers.")

        if isinstance(range, Number):
            self.range = (-range, range)
        else:
            self.range = range

    def sample(self, shape):
        return floatX(np.random.uniform(
            low=self.range[0], high=self.range[1], size=shape))


class GlorotUniform(Uniform):
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

    def sample(self, shape):
        # This code makes some assumptions about the meanings of
        # the different dimensions, which hold for
        # layers.DenseLayer and layers.Conv*DLayer, but not
        # necessarily for other layer types.
        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])
        m = self.gain * np.sqrt(6.0 / ((n1 + n2) * receptive_field_size))
        return floatX(np.random.uniform(low=-m, high=m, size=shape))


class GlorotUniform_c01b(GlorotUniform):
    def sample(self, shape):
        if len(shape) != 4:
            raise RuntimeError(
                "This initializer only works with shapes of length 4")

        n1, n2 = shape[0], shape[3]
        receptive_field_size = shape[1] * shape[2]
        m = self.gain * np.sqrt(6.0 / ((n1 + n2) * receptive_field_size))
        return floatX(np.random.uniform(low=-m, high=m, size=shape))


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
            raise RuntimeError(
                "sparse initializer only works with shapes of length 2")

        w = floatX(np.zeros(shape))
        n_inputs, n_outputs = shape
        size = int(self.sparsity * n_inputs)  # fraction of number of inputs

        for k in range(n_outputs):
            indices = np.arange(n_inputs)
            np.random.shuffle(indices)
            indices = indices[:size]
            values = floatX(np.random.normal(0.0, self.std, size=size))
            w[indices, k] = values

        return w


class Orthogonal(Initializer):
    """
    Orthogonal matrix initialization. For n-dimensional shapes where n > 2,
    the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.
    """
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

    def sample(self, shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return floatX(self.gain * q)
