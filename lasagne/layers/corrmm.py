import theano

from .. import init
from .. import nonlinearities

from .base import Layer

from .conv import conv_output_length
from ..utils import as_tuple

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.blas import GpuCorrMM


__all__ = [
    "MMLayer",
    "Conv2DMMLayer",
]


if not theano.config.device.startswith("gpu"):
    raise ImportError("requires a GPU to work")


# base class for all layers that rely on GpuCorrMM directly
class MMLayer(Layer):
    pass


class Conv2DMMLayer(MMLayer):
    """
    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.  This is an alternative implementation
    which uses ``theano.sandbox.cuda.blas.GpuCorrMM`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_height, input_width)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or tuple of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or tuple of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    border_mode : str, one of 'valid', 'full', 'same'
        A string indicating the convolution border mode.

        If 'valid', the convolution is only computed where the input and the
        filter fully overlap.

        If 'full', the convolution is computed wherever the input and the
        filter overlap by at least one position.

        If 'same', the convolution is computed wherever the input and the
        filter overlap by at least half the filter size, when the filter size
        is odd. In practice, the input is zero-padded with half the filter size
        at the beginning and half at the end (or one less than half in the case
        of an even filter size). This results in an output length that is the
        same as the input length (for both odd and even filter sizes).

    untie_biases : bool, default False
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. This should initialize the
        layer weights to a 4D array with shape
        ``(num_filters, num_input_channels, filter_height, filter_width)``.
        See :meth:`Layer.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If None is provided, the
        layer will have no biases. This should initialize the layer biases to
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, input_height, input_width)`` instead.
        See :meth:`Layer.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    pad : int, tuple of int or None
        An integer or a 2-element tuple specifying the amount of zero-padding
        on each side. This may also be ``None``, in which case the correct
        amount of padding will be inferred from the specified ``border_mode``.

    flip_filters : bool, default False
        Whether to flip the filters and perform a convolution, or not to flip
        them and perform a correlation. Flipping adds a bit of overhead, so it
        is disabled by default. In most cases this does not make a difference
        anyway because the filters are learnt. However, ``flip_filters`` should
        be set to ``True`` if weights are loaded into it that were learnt using
        a regular :class:`lasagne.layers.Conv2DLayer`, for example.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable
        Variable representing the filter weights.

    b : Theano shared variable
        Variable representing the biases.

    Notes
    -----
    Unlike :class:`lasagne.layers.Conv2DLayer`, this layer properly supports
    the 'same' border mode. It is not emulated. This should result in better
    performance.

    Only one of ``pad`` and ``border_mode`` should be specified.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 border_mode=None, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 pad=None, flip_filters=False, **kwargs):
        super(Conv2DMMLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and "
                               "'pad'. To avoid ambiguity, please specify "
                               "only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0, 0)
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0, 0)
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] - 1)
            elif border_mode == 'same':
                # only works for odd filter size, but the even filter size case
                # is probably not worth supporting.
                self.pad = ((self.filter_size[0] - 1) // 2,
                            (self.filter_size[1] - 1) // 2)
            else:
                raise RuntimeError("Unsupported border_mode for "
                                   "Conv2DMMLayer: %s" % border_mode)
        else:
            self.pad = as_tuple(pad, 2)

        self.W = self.create_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        elif self.untie_biases:
            self.b = self.create_param(b, (num_filters, self.output_shape[2],
                                           self.output_shape[3]), name="b")
        else:
            self.b = self.create_param(b, (num_filters,), name="b")

        self.corr_mm_op = GpuCorrMM(subsample=self.stride, pad=self.pad)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         'pad', self.pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            'pad', self.pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        filters = self.W
        if self.flip_filters:
            filters = filters[:, :, ::-1, ::-1]  # flip top-down, left-right

        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)
        conved = self.corr_mm_op(contiguous_input, contiguous_filters)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)
