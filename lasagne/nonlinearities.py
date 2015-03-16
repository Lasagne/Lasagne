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
def rectify(x):
    # The following is faster than T.maximum(0, x),
    # and it works with nonsymbolic inputs as well.
    # Thanks to @SnipyHollow for pointing this out. Also see:
    # http://github.com/benanne/Lasagne/pull/163#issuecomment-81765117
    return 0.5 * (x + abs(x))


# leaky rectify
# Maas et al: Rectifier Nonlinearities Improve Neural Network Acoustic Models
# http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
class LeakyRectify(object):

    def __init__(self, leakiness=0.01):
        self.leakiness = leakiness

    def __call__(self, x):
        if self.leakiness:
            # The following is faster than T.maximum(leakiness * x, x),
            # and it works with nonsymbolic inputs as well. Also see:
            # http://github.com/benanne/Lasagne/pull/163#issuecomment-81765117
            f1 = 0.5 * (1 + self.leakiness)
            f2 = 0.5 * (1 - self.leakiness)
            return f1 * x + f2 * abs(x)
        else:
            return rectify(x)


leaky_rectify = LeakyRectify()  # shortcut with default leakiness


# linear
def linear(x):
    return x

identity = linear
