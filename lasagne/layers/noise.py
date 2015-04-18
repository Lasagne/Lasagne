import numpy as np
import theano
import theano.tensor as T

from .base import Layer

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_srng = RandomStreams()


__all__ = [
    "DropoutLayer",
    "dropout",
    "GaussianNoiseLayer",
]


class DropoutLayer(Layer):
    """Dropout layer [1,2]

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    input_layer : `Layer` instance
        The layer from which this layer will obtain its input
    p : float or scalar tensor
        The probability of setting a value to zero

    Notes
    ----------
    The dropout layer is a regularizer that randomly sets input values to
    zero, see [1,2] for a why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see [1,2] for further discussion. Note that this
    implementation scales the input at training time.

    References
    ----------
    [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
    Salakhutdinov, R. R. (2012).
    Improving neural networks by preventing co-adaptation of feature detectors.
    arXiv Preprint, 1207.0580(Hinton, Geoffrey E., et al.

    [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
    I., & Salakhutdinov, R. R. (2014).
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
    Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, input_layer, p=0.5, rescale=True):
        super(DropoutLayer, self).__init__(input_layer)
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, use full at test time
        rescale: bool
            If true the input is rescaled with input / (1-p) when deterministic
            is False.
        """
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_layer.get_output_shape()
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * _srng.binomial(input_shape, p=retain_prob,
                                          dtype=theano.config.floatX)

dropout = DropoutLayer # shortcut


class GaussianNoiseLayer(Layer):
    """Gaussian noise layer [1]

    Add zero Guassian noise with mean 0 and std. sigma to the input

    Parameters
    ----------
    input_layer : `Layer` instance
        The layer from which this layer will obtain its input
    sigma : float or tensor scalar
        Std. of added gaussian noise

    Notes
    ----------
    The Guassian noise layer is a regularizer. During training you should set
    deterministic to false and during testing you should set deterministic to
    true.

    References
    ----------
    [1] K.-C. Jim, C. Giles, and B. Horne.
    An analysis of noise in recurrent neural networks: convergence and
    generalization.
    Neural Networks, IEEE Trans- actions on, 7(6):1424-1438, 1996.

    """
    def __init__(self, input_layer, sigma=0.1):
        super(GaussianNoiseLayer, self).__init__(input_layer)
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, use full at test time
        rescale: bool
            If true the input is rescaled with input / (1-p) when deterministic
            is False.
        """
        if deterministic or self.sigma == 0:
            return input
        else:
            return input + _srng.normal(input.shape, avg=0.0, std=self.sigma)