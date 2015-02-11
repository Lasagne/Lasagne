
import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities
from ..theano_extensions import conv

from .base import Layer


__all__ = [
    "LocalResponseNormalization2DLayer",
]

class LocalResponseNormalization2DLayer(Layer):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    Input order is assumed to be `BC01`.

    If the value of the ith channel is :math:`x_i`, the output is

    .. math::

        x_i = \frac{x_i}{ (k + ( \alpha \sum_j x_j^2 ))^\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    This code is adapted from pylearn2.
    """

    def __init__(self, incoming, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        """
        :parameters:
            - incoming: input layer or shape
            - alpha: see equation above
            - k: see equation above
            - beta: see equation above
            - n: number of adjacent channels to normalize over.
        """
        super(LocalResponseNormalization2DLayer, self).__init__(incoming, **kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, input_shape=None, *args, **kwargs):
        if input_shape is None:
            input_shape = self.input_shape
        half_n = self.n // 2
        input_sqr = T.sqr(input)
        b, ch, r, c = input_shape
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:,half_n:half_n+ch,:,:], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:,i:i+ch,:,:]
        scale = scale ** self.beta
        return input / scale
