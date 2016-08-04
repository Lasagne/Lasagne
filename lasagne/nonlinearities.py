# -*- coding: utf-8 -*-
"""
Non-linear activation functions for artificial neurons.
"""

import theano.tensor


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


# scaled tanh
class ScaledTanH(object):
    """Scaled tanh :math:`\\varphi(x) = \\tanh(\\alpha \\cdot x) \\cdot \\beta`

    This is a modified tanh function which allows to rescale both the input and
    the output of the activation.

    Scaling the input down will result in decreasing the maximum slope of the
    tanh and as a result it will be in the linear regime in a larger interval
    of the input space. Scaling the input up will increase the maximum slope
    of the tanh and thus bring it closer to a step function.

    Scaling the output changes the output interval to :math:`[-\\beta,\\beta]`.

    Parameters
    ----------
    scale_in : float32
        The scale parameter :math:`\\alpha` for the input

    scale_out : float32
        The scale parameter :math:`\\beta` for the output

    Methods
    -------
    __call__(x)
        Apply the scaled tanh function to the activation `x`.

    Examples
    --------
    In contrast to other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:

    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import ScaledTanH
    >>> scaled_tanh = ScaledTanH(scale_in=0.5, scale_out=2.27)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=scaled_tanh)

    Notes
    -----
    LeCun et al. (in [1]_, Section 4.4) suggest ``scale_in=2./3`` and
    ``scale_out=1.7159``, which has :math:`\\varphi(\\pm 1) = \\pm 1`,
    maximum second derivative at 1, and an effective gain close to 1.

    By carefully matching :math:`\\alpha` and :math:`\\beta`, the nonlinearity
    can also be tuned to preserve the mean and variance of its input:

      * ``scale_in=0.5``, ``scale_out=2.4``: If the input is a random normal
        variable, the output will have zero mean and unit variance.
      * ``scale_in=1``, ``scale_out=1.6``: Same property, but with a smaller
        linear regime in input space.
      * ``scale_in=0.5``, ``scale_out=2.27``: If the input is a uniform normal
        variable, the output will have zero mean and unit variance.
      * ``scale_in=1``, ``scale_out=1.48``: Same property, but with a smaller
        linear regime in input space.

    References
    ----------
    .. [1] LeCun, Yann A., et al. (1998):
       Efficient BackProp,
       http://link.springer.com/chapter/10.1007/3-540-49430-8_2,
       http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    .. [2] Masci, Jonathan, et al. (2011):
       Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction,
       http://link.springer.com/chapter/10.1007/978-3-642-21735-7_7,
       http://people.idsia.ch/~ciresan/data/icann2011.pdf
    """

    def __init__(self, scale_in=1, scale_out=1):
        self.scale_in = scale_in
        self.scale_out = scale_out

    def __call__(self, x):
        return theano.tensor.tanh(x * self.scale_in) * self.scale_out


ScaledTanh = ScaledTanH  # alias with alternative capitalization


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
    return theano.tensor.nnet.relu(x)


# leaky rectify
class LeakyRectify(object):
    """Leaky rectifier :math:`\\varphi(x) = \\max(\\alpha \\cdot x, x)`

    The leaky rectifier was introduced in [1]_. Compared to the standard
    rectifier :func:`rectify`, it has a nonzero gradient for negative input,
    which often helps convergence.

    Parameters
    ----------
    leakiness : float
        Slope for negative input, usually between 0 and 1.
        A leakiness of 0 will lead to the standard rectifier,
        a leakiness of 1 will lead to a linear activation function,
        and any value in between will give a leaky rectifier.

    Methods
    -------
    __call__(x)
        Apply the leaky rectify function to the activation `x`.

    Examples
    --------
    In contrast to other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:

    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import LeakyRectify
    >>> custom_rectify = LeakyRectify(0.1)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=custom_rectify)

    Alternatively, you can use the provided instance for leakiness=0.01:

    >>> from lasagne.nonlinearities import leaky_rectify
    >>> l2 = DenseLayer(l_in, num_units=200, nonlinearity=leaky_rectify)

    Or the one for a high leakiness of 1/3:

    >>> from lasagne.nonlinearities import very_leaky_rectify
    >>> l3 = DenseLayer(l_in, num_units=200, nonlinearity=very_leaky_rectify)

    See Also
    --------
    leaky_rectify: Instance with default leakiness of 0.01, as in [1]_.
    very_leaky_rectify: Instance with high leakiness of 1/3, as in [2]_.

    References
    ----------
    .. [1] Maas et al. (2013):
       Rectifier Nonlinearities Improve Neural Network Acoustic Models,
       http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    .. [2] Graham, Benjamin (2014):
       Spatially-sparse convolutional neural networks,
       http://arxiv.org/abs/1409.6070
    """
    def __init__(self, leakiness=0.01):
        self.leakiness = leakiness

    def __call__(self, x):
        return theano.tensor.nnet.relu(x, self.leakiness)


leaky_rectify = LeakyRectify()  # shortcut with default leakiness
leaky_rectify.__doc__ = """leaky_rectify(x)

    Instance of :class:`LeakyRectify` with leakiness :math:`\\alpha=0.01`
    """


very_leaky_rectify = LeakyRectify(1./3)  # shortcut with high leakiness
very_leaky_rectify.__doc__ = """very_leaky_rectify(x)

     Instance of :class:`LeakyRectify` with leakiness :math:`\\alpha=1/3`
     """


# elu
def elu(x):
    """Exponential Linear Unit :math:`\\varphi(x) = (x > 0) ? x : e^x - 1`

    The Exponential Linear Unit (ELU) was introduced in [1]_. Compared to the
    linear rectifier :func:`rectify`, it has a mean activation closer to zero
    and nonzero gradient for negative input, which can help convergence.
    Compared to the leaky rectifier :class:`LeakyRectify`, it saturates for
    highly negative inputs.

    Parameters
    ----------
    x : float32
        The activation (the summed, weighed input of a neuron).

    Returns
    -------
    float32
        The output of the exponential linear unit for the activation.

    Notes
    -----
    In [1]_, an additional parameter :math:`\\alpha` controls the (negative)
    saturation value for negative inputs, but is set to 1 for all experiments.
    It is omitted here.

    References
    ----------
    .. [1] Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015):
       Fast and Accurate Deep Network Learning by Exponential Linear Units
       (ELUs), http://arxiv.org/abs/1511.07289
    """
    return theano.tensor.switch(x > 0, x, theano.tensor.exp(x) - 1)


# softplus
def softplus(x):
    """Softplus activation function :math:`\\varphi(x) = \\log(1 + e^x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the softplus function applied to the activation.
    """
    return theano.tensor.nnet.softplus(x)


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
