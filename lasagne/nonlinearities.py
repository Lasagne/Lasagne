"""
Nonlinearities
"""

# sigmoid
from theano.tensor.nnet import sigmoid

# softmax (row-wise)
from theano.tensor.nnet import softmax

# tanh
from theano.tensor import tanh


# rectify
# The following is faster than lambda x: T.maximum(0, x)
# Thanks to @SnippyHolloW for pointing this out.
# See: https://github.com/SnippyHolloW/abnet/blob/807aeb9/layers.py#L15
def rectify(x):
    return (x + abs(x)) / 2.0


# leaky rectify
# Maas et al: Rectifier Nonlinearities Improve Neural Network Acoustic Models
# http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
class leaky_rectify(object):

    def __init__(self, leakiness=0.01):
        self.leakiness = leakiness

    def __call__(self, x):
        if self.leakiness:
            import theano.tensor as T
            return T.maximum(self.leakiness * x, x)
        else:
            return rectify(x)


# linear
def linear(x):
    return x

identity = linear
