"""
Non-linear activation functions for artificial neurons.
"""

# sigmoid
from theano.tensor.nnet import sigmoid

# softmax (row-wise)
from theano.tensor.nnet import softmax

# tanh
from theano.tensor import tanh


# rectify
def rectify(x):
    """Rectify activation function :math:`\\varphi(x) = \\max(0, x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weigthed input of a neuron).

    Returns
    -------
    float32
        The output of the rectify function applied to the activation.
    """
    # The following is faster than T.maximum(0, x),
    # and it works with nonsymbolic inputs as well.
    # Thanks to @SnipyHollow for pointing this out. Also see:
    # http://github.com/benanne/Lasagne/pull/163#issuecomment-81765117
    return 0.5 * (x + abs(x))


# leaky rectify
class LeakyRectify(object):
    """Implementation of a leaky rectifier.

    Parameters
    ----------
    leakiness : float in [0, 1]
        A leakiness of 0 will lead to the standard rectifier,
        a leakiness of 1 will lead to a linear activation function.

    Methods
    -------
    __call__(x)
        Calculate the neurons output by applying this leaky rectifier to the
        activation `x`.

    References
    ----------
    The leaky rectifier is described in [1]_.

    .. [1] Maas et al, "Rectifier Nonlinearities Improve Neural Network
       Acoustic Models",
       http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

    """
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
    """Linear activation function :math:`\\varphi(x) = x`

    Parameters
    ----------
    x : float32
        The activation (the summed, weigthed input of a neuron).

    Returns
    -------
    float32
        The output of the identity applied to the activation.
    """
    return x

identity = linear
