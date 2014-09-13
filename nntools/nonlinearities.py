"""
Nonlinearities
"""
import theano
import theano.tensor as T


# sigmoid
from theano.tensor.nnet import sigmoid

# softmax (row-wise)
from theano.tensor.nnet import softmax

# tanh
from theano.tensor import tanh

# rectify
# The following is faster than lambda x: T.maximum(0, x)
# Thanks to @SnippyHolloW for pointing this out.
# See: https://github.com/SnippyHolloW/abnet/blob/master/layers.py#L15
rectify = lambda x: (x + abs(x)) / 2.0

# linear
linear = lambda x: x
identity = linear