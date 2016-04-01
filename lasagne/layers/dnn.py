import theano
from theano.sandbox.cuda import dnn

from .. import init
from .. import nonlinearities
from .base import Layer

from .conv import conv_output_length, BaseConvLayer
from .pool import pool_output_length
from ..utils import as_tuple

if not theano.sandbox.cuda.cuda_enabled:
    raise ImportError(
            "requires GPU support -- see http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#gpu-support")  # pragma: no cover
elif not dnn.dnn_available():
    raise ImportError(
            "cuDNN not available: %s\nSee http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#cudnn" %
            dnn.dnn_available.msg)  # pragma: no cover


__all__ = [
    "Pool2DDNNLayer",
    "MaxPool2DDNNLayer",
    "Pool3DDNNLayer",
    "MaxPool3DDNNLayer",
    "Conv2DDNNLayer",
    "Conv3DDNNLayer",
]


class Pool2DDNNLayer(Layer):
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

    ignore_border : bool (default: True)
        This implementation never includes partial pooling regions, so this
        argument must always be set to True. It exists only to make sure the
        interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.

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
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2DDNNLayer, self).__init__(incoming, **kwargs)
        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))
        self.pool_size = as_tuple(pool_size, 2)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)
        self.pad = as_tuple(pad, 2)
        self.mode = mode
        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=False is not supported. Borders are always ignored.
        if not ignore_border:
            raise NotImplementedError("Pool2DDNNLayer does not support "
                                      "ignore_border=False.")

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=True,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=True,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return dnn.dnn_pool(input, self.pool_size, self.stride,
                            self.mode, self.pad)


class MaxPool2DDNNLayer(Pool2DDNNLayer):
    """
    2D max-pooling layer

    Subclass of :class:`Pool2DDNNLayer` fixing ``mode='max'``, provided for
    compatibility to other ``MaxPool2DLayer`` classes.
    """
    def __init__(self, incoming, pool_size, stride=None,
                 pad=(0, 0), ignore_border=True, **kwargs):
        super(MaxPool2DDNNLayer, self).__init__(incoming, pool_size, stride,
                                                pad, ignore_border, mode='max',
                                                **kwargs)



class Pool3DDNNLayer(Layer):
    """
    3D pooling layer

    Performs 3D mean- or max-pooling over the 3 trailing axes of a 5D input
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

    ignore_border : bool (default: True)
        This implementation never includes partial pooling regions, so this
        argument must always be set to True. It exists only to make sure the
        interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.

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

    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool3DDNNLayer, self).__init__(incoming, **kwargs)
        if len(self.input_shape) != 5:
            raise ValueError("Tried to create a 3D pooling layer with "
                             "input shape %r. Expected 5 input dimensions "
                             "(batchsize, channels, 3 spatial dimensions)."
                             % (self.input_shape,))
        self.pool_size = as_tuple(pool_size, 3)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 3)
        self.pad = as_tuple(pad, 3)
        self.mode = mode
        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=False is not supported. Borders are always ignored.
        if not ignore_border:
            raise NotImplementedError("Pool3DDNNLayer does not support "
                                      "ignore_border=False.")


    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=True,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=True,
                                             )
        
        output_shape[4] = pool_output_length(input_shape[4],
                                             pool_size=self.pool_size[2],
                                             stride=self.stride[2],
                                             pad=self.pad[2],
                                             ignore_border=True,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return dnn.dnn_pool(input, self.pool_size, self.stride,
                            self.mode, self.pad)


class MaxPool3DDNNLayer(Pool3DDNNLayer):
    """
    3D max-pooling layer

    Subclass of :class:`Pool3DDNNLayer` fixing ``mode='max'``, provided for
    consistency to ``MaxPool2DLayer`` classes.
    """
    def __init__(self, incoming, pool_size, stride=None,
                 pad=(0, 0, 0), ignore_border=True, **kwargs):
        super(MaxPool3DDNNLayer, self).__init__(incoming, pool_size, stride,
                                                pad, ignore_border, mode='max',
                                                **kwargs)


class Conv2DDNNLayer(BaseConvLayer):
    """
    lasagne.layers.Conv2DDNNLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
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

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: False)
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
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=False, **kwargs):
        super(Conv2DDNNLayer, self).__init__(incoming, num_filters,
                                             filter_size, stride, pad,
                                             untie_biases, W, b, nonlinearity,
                                             flip_filters, n=2, **kwargs)

    def convolve(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)

        conved = dnn.dnn_conv(img=input,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=border_mode,
                              conv_mode=conv_mode
                              )
        return conved


class Conv3DDNNLayer(BaseConvLayer):
    """
    lasagne.layers.Conv3DDNNLayer(incoming, num_filters, filter_size,
    stride=(1, 1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
    **kwargs)

    3D convolutional layer

    Performs a 3D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.  This implementation uses
    ``theano.sandbox.cuda.dnn.dnn_conv3d`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 5D tensor, with shape ``(batch_size,
        num_input_channels, input_rows, input_columns, input_depth)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 3-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 3-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of three integers allows different symmetric
        padding per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        4D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 5D tensor with shape ``(num_filters,
        num_input_channels, filter_rows, filter_columns, filter_depth)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns, output_depth)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: False)
        Whether to flip the filters and perform a convolution, or not to flip
        them and perform a correlation. Flipping adds a bit of overhead, so it
        is disabled by default. In most cases this does not make a difference
        anyway because the filters are learned, but if you want to compute
        predictions with pre-trained weights, take care if they need flipping.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=False, **kwargs):
        super(Conv3DDNNLayer, self).__init__(incoming, num_filters,
                                             filter_size, stride, pad,
                                             untie_biases, W, b, nonlinearity,
                                             flip_filters, n=3, **kwargs)

    def convolve(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)

        conved = dnn.dnn_conv3d(img=input,
                                kerns=self.W,
                                subsample=self.stride,
                                border_mode=border_mode,
                                conv_mode=conv_mode
                                )
        return conved
