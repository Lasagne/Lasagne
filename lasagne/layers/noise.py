import theano
import numpy as np

from .base import Layer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "DropoutLayer",
    "dropout",
    "GaussianNoiseLayer",
]


class DropoutLayer(Layer):
    """Dropout layer [1]_, [2]_

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero, see references for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=theano.config.floatX)

dropout = DropoutLayer  # shortcut


class GaussianNoiseLayer(Layer):
    """Gaussian noise layer [1]_.

    Add zero Gaussian noise with mean 0 and std sigma to the input

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape
    sigma : float or tensor scalar
            Std of added Gaussian noise

    Notes
    -----
    The Gaussian noise layer is a regularizer. During training you should set
    deterministic to false and during testing you should set deterministic to
    true.

    References
    ----------
    .. [1] K.-C. Jim, C. Giles, and B. Horne (1996):
           An analysis of noise in recurrent neural networks: convergence and
           generalization.
           IEEE Transactions on Neural Networks, 7(6):1424-1438.
    """
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        if deterministic or self.sigma == 0:
            return input
        else:
            return input + self._srng.normal(input.shape,
                                             avg=0.0,
                                             std=self.sigma)
