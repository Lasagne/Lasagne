import theano
from theano.sandbox.cuda import dnn

from .. import init
from .. import nonlinearities
from .base import Layer

from .conv import conv_output_length
from .pool import pool_output_length
from ..utils import as_tuple

if not theano.config.device.startswith("gpu") or not dnn.dnn_available():
    raise ImportError("dnn not available")  # pragma: no cover


__all__ = [
    "Pool2DDNNLayer",
    "MaxPool2DDNNLayer",
    "Conv2DDNNLayer",
]


class DNNLayer(Layer):
    pass


class Pool2DDNNLayer(DNNLayer):
    """
    2D pooling layer

    Performs 2D mean- or max-pooling over the two trailing axes of a 4D input
    tensor. This is an alternative implementation which uses
    ``theano.sandbox.cuda.dnn.dnn_pool`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension. If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    mode : string
        Pooling mode, one of 'max', 'average_inc_pad' or 'average_exc_pad'.
        Defaults to 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    This is a drop-in replacement for :class:`lasagne.layers.MaxPool2DLayer`.
    Its interface is the same, except it does not support the ``ignore_border``
    argument.
    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 mode='max', **kwargs):
        super(Pool2DDNNLayer, self).__init__(incoming, **kwargs)
        self.pool_size = as_tuple(pool_size, 2)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)
        self.pad = as_tuple(pad, 2)
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             ignore_border=True,
                                             pad=self.pad[0],
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             ignore_border=True,
                                             pad=self.pad[1],
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return dnn.dnn_pool(input, self.pool_size, self.stride,
                            self.mode, self.pad)


class MaxPool2DDNNLayer(Pool2DDNNLayer):  # for consistency
    def __init__(self, incoming, pool_size, stride=None,
                 pad=(0, 0), **kwargs):
        super(MaxPool2DDNNLayer, self).__init__(incoming, pool_size, stride,
                                                pad, mode='max', **kwargs)


class Conv2DDNNLayer(DNNLayer):
    """
    lasagne.layers.Conv2DDNNLayer(incoming, num_filters, filter_size,
    stride=(1, 1), border_mode=None, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, pad=None, flip_filters=False,
    **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.  This is an alternative implementation
    which uses ``theano.sandbox.cuda.dnn.dnn_conv`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable
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
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If None is provided, the
        layer will have no biases. This should initialize the layer biases to
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, input_rows, input_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    pad : int, iterable or None
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
        super(Conv2DDNNLayer, self).__init__(incoming, **kwargs)
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
            self.border_mode = 'valid'
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0, 0)
                self.border_mode = 'valid'
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] - 1)
                self.border_mode = 'full'
            elif border_mode == 'same':
                # dnn_conv does not support same, so we just specify
                # padding directly.
                # only works for odd filter size, but the even filter size
                # case is probably not worth supporting.
                self.pad = ((self.filter_size[0] - 1) // 2,
                            (self.filter_size[1] - 1) // 2)
                self.border_mode = None
            else:
                raise RuntimeError("Invalid border mode: '%s'" % border_mode)
        else:
            self.pad = as_tuple(pad, 2)
            self.border_mode = None

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2],
                                self.output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

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
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        # if 'border_mode' is one of 'valid' or 'full' use that.
        # else use pad directly.
        border_mode = (self.border_mode if (self.border_mode is not None)
                       else self.pad)

        conved = dnn.dnn_conv(img=input,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=border_mode,
                              conv_mode=conv_mode
                              )

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)
