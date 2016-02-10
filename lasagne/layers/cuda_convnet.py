import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer

from .conv import conv_output_length, BaseConvLayer
from .pool import pool_output_length
from ..utils import as_tuple

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs

__all__ = [
    "Conv2DCCLayer",
    "MaxPool2DCCLayer",
    "ShuffleBC01ToC01BLayer",
    "bc01_to_c01b",
    "ShuffleC01BToBC01Layer",
    "c01b_to_bc01",
    "NINLayer_c01b",
]


if not theano.sandbox.cuda.cuda_enabled:
    raise ImportError(
            "requires GPU support -- see http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#gpu-support")  # pragma: no cover


class Conv2DCCLayer(BaseConvLayer):
    """
    lasagne.layers.Conv2DCCLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False, W=None,
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
    dimshuffle=True, flip_filters=False, partial_sum=1, **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.  This is an alternative implementation
    which uses the cuda-convnet wrappers from pylearn2:
    ``pylearn2.sandbox.cuda_convnet.filter_acts.FilterActs``.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. This
        layer expects a 4D tensor as its input, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
        If automatic dimshuffling is disabled (see notes), the shape should be
        ``(num_input_channels, input_rows, input_columns, batch_size)``
        instead (c01b axis order).

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.
        This layer does not support non-square filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation. This layer does not support using different
        strides along both axes.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders. This layer does not support using different amounts of
        padding along both axes, but for compatibility to other layers you can
        still specify the padding as a tuple of two same-valued integers.

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

        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        If automatic dimshuffling is disabled (see notes), the shape should be
        ``(num_input_channels, input_rows, input_columns, num_filters)``
        instead (c01b axis order).
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

    dimshuffle : bool (default: True)
        If ``True``, the layer will automatically apply the necessary
        dimshuffle operations to deal with the fact that the cuda-convnet
        implementation uses c01b (batch-size-last) axis order instead of bc01
        (batch-size-first), which is the Lasagne/Theano default. This makes the
        layer interoperable with other Lasagne layers.

        If ``False``, this automatic dimshuffling is disabled and the layer
        will expect its input and parameters to have c01b axis order. It is up
        to the user to ensure this. :class:`ShuffleBC01ToC01BLayer` and
        :class:`ShuffleC01BToBC01Layer` can be used to convert between bc01 and
        c01b axis order.

    flip_filters : bool (default: False)
        Whether to flip the filters and perform a convolution, or not to flip
        them and perform a correlation. Flipping adds a bit of overhead, so it
        is disabled by default. In most cases this does not make a difference
        anyway because the filters are learnt. However, ``flip_filters`` should
        be set to ``True`` if weights are loaded into it that were learnt using
        a regular :class:`lasagne.layers.Conv2DLayer`, for example.

    partial_sum : int or None (default: 1)
        This value tunes the trade-off between memory usage and performance.
        You can specify any positive integer that is a divisor of the output
        feature map size (i.e. output rows times output columns). Higher
        values decrease memory usage, but also performance. Specifying 0 or
        ``None`` means the highest possible value will be used. The Lasagne
        default of ``1`` gives the best performance, but also the highest
        memory usage.

        More information about this parameter can be found in the
        `cuda-convnet documentation
        <https://code.google.com/p/cuda-convnet/wiki/LayerParams>`_.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    Notes
    -----
    The cuda-convnet convolution implementation has several limitations:

    * only square filters are supported.
    * only identical strides in the horizontal and vertical direction are
      supported.
    * the number of filters must be a multiple of 16.
    * the number of input channels must be even, or less than or equal to
      3.
    * if the gradient w.r.t. the input is to be computed, the number of
      channels must be divisible by 4.
    * performance is optimal when the batch size is a multiple of 128 (but
      other batch sizes are supported).
    * this layer only works on the GPU.

    The cuda-convnet convolution implementation uses c01b (batch-size-last)
    axis order by default. The Theano/Lasagne default is bc01
    (batch-size-first). This layer automatically adds the necessary dimshuffle
    operations for the input and the parameters so that it is interoperable
    with other layers that assume bc01 axis order. However, these additional
    dimshuffle operations may sometimes negatively affect performance. For this
    reason, it is possible to disable them by setting ``dimshuffle=False``. In
    this case, the user is expected to manually ensure that the input and
    parameters have the correct axis order. :class:`ShuffleBC01ToC01BLayer` and
    :class:`ShuffleC01BToBC01Layer` can be used to convert between bc01 and
    c01b axis order.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=None,
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 dimshuffle=True, flip_filters=False, partial_sum=1,
                 **kwargs):
        if W is None:
            if dimshuffle:
                W = init.GlorotUniform()
            else:
                W = init.GlorotUniform(c01b=True)
        self.dimshuffle = dimshuffle

        super(Conv2DCCLayer, self).__init__(incoming, num_filters, filter_size,
                                            stride, pad, untie_biases, W, b,
                                            nonlinearity, flip_filters, n=2,
                                            **kwargs)
        self.partial_sum = partial_sum

        if self.filter_size[0] != self.filter_size[1]:
            raise RuntimeError("Conv2DCCLayer only supports square filters, "
                               "but filter_size=(%d, %d)" % filter_size)

        if self.stride[0] != self.stride[1]:
            raise RuntimeError("Conv2DCCLayer only supports square strides, "
                               "but stride=(%d, %d)" % stride)

        if self.num_filters % 16 != 0:
            raise RuntimeError("Conv2DCCLayer requires num_filters to be a "
                               "multiple of 16, but num_filters is "
                               "%d" % num_filters)

        if not (self.num_input_channels < 4 or
                self.num_input_channels % 4 == 0):
            raise RuntimeError("Conv2DCCLayer requires the number of input "
                               "channels to be 1, 2, 3 or a multiple of 4, "
                               "but it is %d" % self.num_input_channels)

        if isinstance(self.pad, tuple):
            if self.pad[0] != self.pad[1]:
                raise RuntimeError("Conv2DCCLayer only supports square "
                                   "padding, but pad=(%d, %d)" % pad)
            pad = self.pad[0]
        elif self.pad == 'same':
            pad = self.filter_size[0] // 2
        elif self.pad == 'full':
            pad = self.filter_size[0] - 1

        if not self.dimshuffle and self.untie_biases and self.b is not None:
            del self.params[self.b]
            biases_shape = (num_filters, self.output_shape[1],
                            self.output_shape[2])
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

        self.filter_acts_op = FilterActs(stride=self.stride[0],
                                         partial_sum=self.partial_sum,
                                         pad=pad)

    @property
    def num_input_channels(self):
        if self.dimshuffle:
            return self.input_shape[1]
        else:
            return self.input_shape[0]

    def get_W_shape(self):
        if self.dimshuffle:
            return super(Conv2DCCLayer, self).get_W_shape()
        else:
            return ((self.num_input_channels,) +
                    self.filter_size +
                    (self.num_filters,))

    def get_output_shape_for(self, input_shape):
        if not self.dimshuffle:
            # c01b to bc01
            input_shape = (input_shape[3], input_shape[0],
                           input_shape[1], input_shape[2])
        shape = super(Conv2DCCLayer, self).get_output_shape_for(input_shape)
        if not self.dimshuffle:
            # bc01 to c01b
            shape = (shape[1], shape[2], shape[3], shape[0])
        return shape

    def get_output_for(self, input, **kwargs):
        if self.dimshuffle:
            filters = self.W.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            input = input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
        else:
            filters = self.W

        if self.flip_filters:
            filters = filters[:, ::-1, ::-1, :]  # flip top-down, left-right

        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.stride != 1:
            # cuda-convnet calculates a non-standard strided output shape,
            # so we need to truncate the output in this case
            pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2
            true_rows = conv_output_length(input.shape[1],
                                           self.filter_size[0],
                                           self.stride[0],
                                           pad[0])
            true_columns = conv_output_length(input.shape[2],
                                              self.filter_size[1],
                                              self.stride[1],
                                              pad[1])
            conved = conved[:, :true_rows, :true_columns, :]

        if self.b is not None:
            if self.untie_biases:
                biases = self.b.dimshuffle(0, 1, 2, 'x')  # c01 to c01b
            else:
                biases = self.b.dimshuffle(0, 'x', 'x', 'x')  # c to c01b
            conved += biases

        conved = self.nonlinearity(conved)

        if self.dimshuffle:
            return conved.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        else:
            return conved


class MaxPool2DCCLayer(Layer):
    """
    2D max-pooling layer

    Performs 2D max-pooling over the two trailing axes of a 4D input tensor
    (or over axis 1 and 2 if ``dimshuffle=False``, see notes). This is an
    alternative implementation which uses the cuda-convnet wrappers from
    pylearn2: ``pylearn2.sandbox.cuda_convnet.pool.MaxPool``.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements. This layer does not support non-square pooling regions.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``. This layer does not support
        using different strides along both axes.

    pad : integer or iterable (default: 0)
        This implementation does not support custom padding, so this argument
        must always be set to ``0``. It exists only to make sure the
        interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.

    ignore_border : bool (default: False)
        This implementation always includes partial pooling regions, so this
        argument must always be set to False. It exists only to make sure the
        interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.

    dimshuffle : bool (default: True)
        If ``True``, the layer will automatically apply the necessary
        dimshuffle operations to deal with the fact that the cuda-convnet
        implementation uses c01b (batch-size-last) axis order instead of bc01
        (batch-size-first), which is the Lasagne/Theano default. This makes the
        layer interoperable with other Lasagne layers.

        If ``False``, this automatic dimshuffling is disabled and the layer
        will expect its input and parameters to have c01b axis order. It is up
        to the user to ensure this. :class:`ShuffleBC01ToC01BLayer` and
        :class:`ShuffleC01BToBC01Layer` can be used to convert between bc01 and
        c01b axis order.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The cuda-convnet max-pooling implementation has several limitations:

    * only square pooling regions are supported.
    * only identical strides in the horizontal and vertical direction are
      supported.
    * only square inputs are supported. (This limitation does not exist for
      the convolution implementation.)
    * partial pooling regions are always included (``ignore_border`` is forced
      to ``False``).
    * custom padding is not supported (``pad`` is forced to ``0``).
    * this layer only works on the GPU.

    The cuda-convnet pooling implementation uses c01b (batch-size-last)
    axis order by default. The Theano/Lasagne default is bc01
    (batch-size-first). This layer automatically adds the necessary dimshuffle
    operations for the input and the parameters so that it is interoperable
    with other layers that assume bc01 axis order. However, these additional
    dimshuffle operations may sometimes negatively affect performance. For this
    reason, it is possible to disable them by setting ``dimshuffle=False``. In
    this case, the user is expected to manually ensure that the input and
    parameters have the correct axis order. :class:`ShuffleBC01ToC01BLayer` and
    :class:`ShuffleC01BToBC01Layer` can be used to convert between bc01 and
    c01b axis order.
    """
    def __init__(self, incoming, pool_size, stride=None, ignore_border=False,
                 dimshuffle=True, **kwargs):
        from pylearn2.sandbox.cuda_convnet.pool import MaxPool

        if 'pad' in kwargs:
            pad = kwargs.pop('pad')
            if as_tuple(pad, 2) != (0, 0):
                raise NotImplementedError("MaxPool2DCCLayer does not "
                                          "support padding")

        super(MaxPool2DCCLayer, self).__init__(incoming, **kwargs)

        pool_size = as_tuple(pool_size, 2)

        if pool_size[0] != pool_size[1]:
            raise NotImplementedError("MaxPool2DCCLayer only supports square "
                                      "pooling regions, but pool_size=(%d, %d)"
                                      % pool_size)

        self.pool_size = pool_size[0]

        if stride is None:
            self.stride = self.pool_size
        else:
            stride = as_tuple(stride, 2)
            if stride[0] != stride[1]:
                raise NotImplementedError("MaxPool2DCCLayer only supports "
                                          "using the same stride in both "
                                          "directions but stride=(%d, %d)"
                                          % stride)
            self.stride = stride[0]

        if self.stride > self.pool_size:
            raise NotImplementedError("MaxPool2DCCLayer only supports "
                                      "stride <= pool_size.")

        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=True is not supported. Borders are never ignored.
        if ignore_border:
            raise NotImplementedError("MaxPool2DCCLayer does not support "
                                      "ignore_border=True.")

        self.dimshuffle = dimshuffle

        self.pool_op = MaxPool(ds=self.pool_size, stride=self.stride)

    def get_output_shape_for(self, input_shape):
        if self.dimshuffle:
            batch_size = input_shape[0]
            num_input_channels = input_shape[1]
            input_rows, input_columns = input_shape[2:4]
        else:
            batch_size = input_shape[3]
            num_input_channels = input_shape[0]
            input_rows, input_columns = input_shape[1:3]

        output_rows = pool_output_length(input_rows,
                                         pool_size=self.pool_size,
                                         stride=self.stride,
                                         pad=0,
                                         ignore_border=False,
                                         )
        output_columns = pool_output_length(input_columns,
                                            pool_size=self.pool_size,
                                            stride=self.stride,
                                            pad=0,
                                            ignore_border=False,
                                            )

        if self.dimshuffle:
            return (batch_size, num_input_channels, output_rows,
                    output_columns)
        else:
            return (num_input_channels, output_rows, output_columns,
                    batch_size)

    def get_output_for(self, input, **kwargs):
        if self.dimshuffle:
            input = input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b

        contiguous_input = gpu_contiguous(input)
        pooled = self.pool_op(contiguous_input)

        if self.dimshuffle:
            return pooled.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        else:
            return pooled


# Helper classes for switching between bc01 and c01b input formats

class ShuffleBC01ToC01BLayer(Layer):
    """
    shuffle 4D input from bc01 (batch-size-first) order to c01b
    (batch-size-last) order.

    This layer can be used for interoperability between c01b and bc01 layers.
    For example, :class:`MaxPool2DCCLayer` and :class:`Conv2DCCLayer` operate
    in c01b mode when they are created with ``dimshuffle=False``.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(1, 2, 3, 0)

bc01_to_c01b = ShuffleBC01ToC01BLayer  # shortcut


class ShuffleC01BToBC01Layer(Layer):
    """
    shuffle 4D input from c01b (batch-size-last) order to bc01
    (batch-size-first) order.

    This layer can be used for interoperability between c01b and bc01 layers.
    For example, :class:`MaxPool2DCCLayer` and :class:`Conv2DCCLayer` operate
    in c01b mode when they are created with ``dimshuffle=False``.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[3], input_shape[0], input_shape[1], input_shape[2])

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(3, 0, 1, 2)

c01b_to_bc01 = ShuffleC01BToBC01Layer  # shortcut


# c01b versions of other Layer classes

class NINLayer_c01b(Layer):
    """
    lasagne.layers.NINLayer_c01b(incoming, num_units, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    Network-in-network layer with c01b axis ordering.

    This is a c01b version of :class:`lasagne.layers.NINLayer`.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    untie_biases : bool
        If ``False``, the network has a single bias vector similar to a dense
        layer. If ``True``, a separate bias vector is used for each spatial
        position.

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be
        (num_units, num_input_channels).
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the correct shape is determined by the
        untie_biases setting. If untie_biases is ``False``, then the shape
        should be ``(num_units,)``. If untie_biases is ``True`` then the shape
        should be ``(num_units, rows, columns)``. If ``None`` is provided the
        layer will have no biases.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    """
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer_c01b, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[0]

        self.W = self.add_param(W, (num_units, num_input_channels), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_units,) + self.output_shape[1:-1]
            else:
                biases_shape = (num_units,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (self.num_units,) + input_shape[1:]

    def get_output_for(self, input, **kwargs):
        # fc * c01b... = f01b...
        out = T.tensordot(self.W, input, axes=[[1], [0]])

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                bias_axes = range(input.ndim - 1) + ['x']
            else:
                bias_axes = [0] + (['x'] * (input.ndim - 1))
            b_shuffled = self.b.dimshuffle(bias_axes)
            activation = out + b_shuffled

        return self.nonlinearity(activation)
