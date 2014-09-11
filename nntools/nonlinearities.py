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
rectify = lambda x: T.maximum(0, x)

# linear
linear = lambda x: x
identity = linear