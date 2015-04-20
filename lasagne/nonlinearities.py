"""
Non-linear activation functions for artificial neurons.
"""

import theano.tensor.nnet


# sigmoid
def sigmoid(x):
    """Sigmoid activation function :math:`\\varphi(x) = \\frac{1}{1 + e^{-x}}`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 in [0, 1]
        The output of the sigmoid function applied to the activation.
    """
    return theano.tensor.nnet.sigmoid(x)


# softmax (row-wise)
def softmax(x):
    """Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 where the sum of the row is 1 and each single value is in [0, 1]
        The output of the softmax function applied to the activation.
    """
    return theano.tensor.nnet.softmax(x)


# tanh
def tanh(x):
    """Tanh activation function :math:`\\varphi(x) = \\tanh(x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 in [-1, 1]
        The output of the tanh function applied to the activation.
    """
    return theano.tensor.tanh(x)


# rectify
def rectify(x):
    """Rectify activation function :math:`\\varphi(x) = \\max(0, x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the rectify function applied to the activation.
    """
    # The following is faster than T.maximum(0, x),
    # and it works with nonsymbolic inputs as well.
    # Thanks to @SnipyHollow for pointing this out. Also see:
    # https://github.com/Lasagne/Lasagne/pull/163#issuecomment-81765117
    return 0.5 * (x + abs(x))


# leaky rectify
class LeakyRectify(object):
    """Implementation of a leaky rectifier
    :math:`\\varphi(x) = \max(leakiness \cdot x, x)`.

    Parameters
    ----------
    leakiness : float in [0, 1]
        A leakiness of 0 will lead to the standard rectifier,
        a leakiness of 1 will lead to a linear activation function.

    Methods
    -------
    __call__(x)
        Calculate the neuron's output by applying this leaky rectifier to the
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
            # https://github.com/Lasagne/Lasagne/pull/163#issuecomment-81765117
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
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the identity applied to the activation.
    """
    return x

identity = linear
