import theano
import theano.tensor as T

from .base import Layer
from ..random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "DropoutLayer",
    "dropout",
    "dropout_channels",
    "spatial_dropout",
    "dropout_locations",
    "GaussianNoiseLayer",
]


class DropoutLayer(Layer):
    """Dropout layer

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If ``True`` (the default), scale the input by ``1 / (1 - p)`` when
        dropout is enabled, to keep the expected output mean the same.
    shared_axes : tuple of int
        Axes to share the dropout mask over. By default, each value can be
        dropped individually. ``shared_axes=(0,)`` uses the same mask across
        the batch. ``shared_axes=(2, 3)`` uses the same mask across the
        spatial dimensions of 2D feature maps.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.

    The behaviour of the layer depends on the ``deterministic`` keyword
    argument passed to :func:`lasagne.layers.get_output`. If ``True``, the
    layer behaves deterministically, and passes on the input unchanged. If
    ``False`` or not specified, dropout (and possibly scaling) is enabled.
    Usually, you would use ``deterministic=False`` at train time and
    ``deterministic=True`` at test time.

    See also
    --------
    dropout_channels : Drops full channels of feature maps
    spatial_dropout : Alias for :func:`dropout_channels`
    dropout_locations : Drops full pixels or voxels of feature maps

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
    def __init__(self, incoming, p=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            return input * mask

dropout = DropoutLayer  # shortcut


def dropout_channels(incoming, *args, **kwargs):
    """
    Convenience function to drop full channels of feature maps.

    Adds a :class:`DropoutLayer` that sets feature map channels to zero, across
    all locations, with probability p. For convolutional neural networks, this
    may give better results than independent dropout [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    *args, **kwargs
        Any additional arguments and keyword arguments are passed on to the
        :class:`DropoutLayer` constructor, except for `shared_axes`.

    Returns
    -------
    layer : :class:`DropoutLayer` instance
        The dropout layer with `shared_axes` set to drop channels.

    References
    ----------
    .. [1] J. Tompson, R. Goroshin, A. Jain, Y. LeCun, C. Bregler (2014):
           Efficient Object Localization Using Convolutional Networks.
           https://arxiv.org/abs/1411.4280
    """
    ndim = len(getattr(incoming, 'output_shape', incoming))
    kwargs['shared_axes'] = tuple(range(2, ndim))
    return DropoutLayer(incoming, *args, **kwargs)

spatial_dropout = dropout_channels  # alias


def dropout_locations(incoming, *args, **kwargs):
    """
    Convenience function to drop full locations of feature maps.

    Adds a :class:`DropoutLayer` that sets feature map locations (i.e., pixels
    or voxels) to zero, across all channels, with probability p.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    *args, **kwargs
        Any additional arguments and keyword arguments are passed on to the
        :class:`DropoutLayer` constructor, except for `shared_axes`.

    Returns
    -------
    layer : :class:`DropoutLayer` instance
        The dropout layer with `shared_axes` set to drop locations.
    """
    kwargs['shared_axes'] = (1,)
    return DropoutLayer(incoming, *args, **kwargs)


class GaussianNoiseLayer(Layer):
    """Gaussian noise layer.

    Add zero-mean Gaussian noise of given standard deviation to the input [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape
    sigma : float or tensor scalar
            Standard deviation of added Gaussian noise

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
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
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
