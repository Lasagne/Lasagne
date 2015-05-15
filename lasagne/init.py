"""
Functions to create initializers for parameter variables.

Examples
--------
>>> from lasagne.layers import DenseLayer
>>> from lasagne.init import Constant, Glorot
>>> l1 = DenseLayer((100,20), num_units=50, W=GlorotUniform(), b=Constant(0.0))
"""

import numpy as np

from .utils import floatX


class Initializer(object):
    """Base class for parameter tensor initializers.

    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.

    """
    def __call__(self, shape):
        """
        Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`sample()` method.
        """
        return self.sample(shape)

    def sample(self, shape):
        """
        Sample should return a theano.tensor of size shape and data type
        theano.config.floatX.

        Parameters
        -----------
        shape : tuple or int
            Integer or tuple specifying the size of the returned
            matrix.
        returns : theano.tensor
            Matrix of size shape and dtype theano.config.floatX.
        """
        raise NotImplementedError()


class Normal(Initializer):
    """Sample initial weights from the Gaussian distribution.

    Initial weight parameters are sampled from N(mean, std).

    Parameters
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return floatX(np.random.normal(self.mean, self.std, size=shape))


class Uniform(Initializer):
    """Sample initial weights from the uniform distribution.

    Parameters are sampled from U(a, b).

    Parameters
    ----------
    range : float or tuple
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    std : float or None
        If std is a float then the weights are sampled from
        U(mean - np.sqrt(3) * std, mean + np.sqrt(3) * std).
    mean : float
        see std for description.
    """
    def __init__(self, range=0.01, std=None, mean=0.0):
        import warnings
        warnings.warn("The uniform initializer no longer uses Glorot et al.'s "
                      "approach to determine the bounds, but defaults to the "
                      "range (-0.01, 0.01) instead. Please use the new "
                      "GlorotUniform initializer to get the old behavior. "
                      "GlorotUniform is now the default for all layers.")

        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number

        self.range = (a, b)

    def sample(self, shape):
        return floatX(np.random.uniform(
            low=self.range[0], high=self.range[1], size=shape))


class Glorot(Initializer):
    """Glorot weight initialization [1]_.

    This is also known as Xavier initialization.

    Parameters
    ----------
    initializer : lasagne.init.Initializer
        Initializer used to sample the weights, must accept `std` in its
        constructor to sample from a distribution with a given standard
        deviation.
    gain : float or 'relu'
        Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
        units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
        functions may need different factors.
    c01b : bool
        For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
        with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
        the correct fan-in and fan-out.

    References
    ----------
    .. [1] Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.

    Notes
    -----
    For a :class:`DenseLayer`, if ``gain='relu'`` and ``initializer=Uniform``,
    the weights are initialized as

    .. math::
       a &= \\sqrt{\\frac{6}{fan_{in}+fan_{out}}}\\\\
       W &\sim U[-a, a]

    If ``gain=1`` and ``initializer=Normal``, the weights are initialized as

    .. math::
       \\sigma &= \\sqrt{\\frac{2}{fan_{in}+fan_{out}}}\\\\
       W &\sim N(0, \\sigma)

    See Also
    --------
    GlorotNormal  : Shortcut with Gaussian initializer.
    GlorotUniform : Shortcut with uniform initializer.
    """
    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(std=std).sample(shape)


class GlorotNormal(Glorot):
    """Glorot with weights sampled from the Normal distribution.

    See :class:`Glorot` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotNormal, self).__init__(Normal, gain, c01b)


class GlorotUniform(Glorot):
    """Glorot with weights sampled from the Uniform distribution.

    See :class:`Glorot` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotUniform, self).__init__(Uniform, gain, c01b)


class He(Initializer):
    """He weight initialization [1]_.

    Weights are initialized with a standard deviation of
    :math:`\\sigma = gain \\sqrt{\\frac{1}{fan_{in}}}`.

    Parameters
    ----------
    initializer : lasagne.init.Initializer
        Initializer used to sample the weights, must accept `std` in its
        constructor to sample from a distribution with a given standard
        deviation.
    gain : float or 'relu'
        Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
        units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
        functions may need different factors.
    c01b : bool
        For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
        with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
        the correct fan-in and fan-out.

    References
    ----------
    .. [1] Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852.

    See Also
    ----------
    HeNormal  : Shortcut with Gaussian initializer.
    HeUniform : Shortcut with uniform initializer.
    """
    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            fan_in = np.prod(shape[:3])
        else:
            if len(shape) == 2:
                fan_in = shape[0]
            elif len(shape) > 2:
                fan_in = np.prod(shape[1:])
            else:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

        std = self.gain * np.sqrt(1.0 / fan_in)
        return self.initializer(std=std).sample(shape)


class HeNormal(He):
    """He initializer with weights sampled from the Normal distribution.

    See :class:`He` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(HeNormal, self).__init__(Normal, gain, c01b)


class HeUniform(He):
    """He initializer with weights sampled from the Uniform distribution.

    See :class:`He` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(HeUniform, self).__init__(Uniform, gain, c01b)


class Constant(Initializer):
    """Initialize weights with constant value.

    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return floatX(np.ones(shape) * self.val)


class Sparse(Initializer):
    """Initialize weights as sparse matrix.

    Parameters
    ----------
    sparsity : float
        Exact fraction of non-zero values per column. Larger values give less
        sparsity.
    std : float
        Non-zero weights are sampled from N(0, std).
    """
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
    """Intialize weights as Orthogonal matrix.

    Orthogonal matrix initialization. For n-dimensional shapes where n > 2,
    the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.

    Parameters
    ----------
    gain : float or 'relu'
        'relu' gives gain of sqrt(2).
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
