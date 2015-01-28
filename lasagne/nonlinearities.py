"""
Nonlinearities
"""
from __future__ import absolute_import
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
# See: https://github.com/SnippyHolloW/abnet/blob/807aeb98e767eb4e295c6d7d60ff5c9006955e0d/layers.py#L15
def rectify(x):
    return (x + abs(x)) / 2.0

# linear
def linear(x):
    return x

identity = linear
