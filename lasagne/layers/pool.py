import theano.tensor as T

from .base import Layer
from ..utils import as_tuple

__all__ = [
    "MaxPool1DLayer",
    "MaxPool2DLayer",
    "Pool1DLayer",
    "Pool2DLayer",
    "Upscale1DLayer",
    "Upscale2DLayer",
    "Upscale3DLayer",
    "FeaturePoolLayer",
    "FeatureWTALayer",
    "GlobalPoolLayer",
    "SpatialPyramidPoolingLayer",
]


def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """
    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length


def pool_2d(input, **kwargs):
    """
    Wrapper function that calls :func:`theano.tensor.signal.pool_2d` either
    with the new or old keyword argument names expected by Theano.
    """
    try:
        return T.signal.pool.pool_2d(input, **kwargs)
    except TypeError:  # pragma: no cover
        # convert from new to old interface
        kwargs['ds'] = kwargs.pop('ws')
        kwargs['st'] = kwargs.pop('stride')
        kwargs['padding'] = kwargs.pop('pad')
        return T.signal.pool.pool_2d(input, **kwargs)


class Pool1DLayer(Layer):
    """
    1D pooling layer

    Performs 1D mean or max-pooling over the trailing axis
    of a 3D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region. If an iterable, it should have a
        single element.

    stride : integer, iterable or ``None``
        The stride between sucessive pooling regions.
        If ``None`` then ``stride == pool_size``.

    pad : integer or iterable
        The number of elements to be added to the input on each side.
        Must be less than stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    See Also
    --------
    MaxPool1DLayer : Shortcut for max pooling layer.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """
    def __init__(self, incoming, pool_size, stride=None, pad=0,
                 ignore_border=True, mode='max', **kwargs):
        super(Pool1DLayer, self).__init__(incoming, **kwargs)

        if len(self.input_shape) != 3:
            raise ValueError("Tried to create a 1D pooling layer with "
                             "input shape %r. Expected 3 input dimensions "
                             "(batchsize, channels, 1 spatial dimensions)."
                             % (self.input_shape,))

        self.pool_size = as_tuple(pool_size, 1)
        self.stride = self.pool_size if stride is None else as_tuple(stride, 1)
        self.pad = as_tuple(pad, 1)
        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[-1] = pool_output_length(input_shape[-1],
                                              pool_size=self.pool_size[0],
                                              stride=self.stride[0],
                                              pad=self.pad[0],
                                              ignore_border=self.ignore_border,
                                              )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_4d = T.shape_padright(input, 1)

        pooled = pool_2d(input_4d,
                         ws=(self.pool_size[0], 1),
                         stride=(self.stride[0], 1),
                         ignore_border=self.ignore_border,
                         pad=(self.pad[0], 0),
                         mode=self.mode,
                         )
        return pooled[:, :, :, 0]


class Pool2DLayer(Layer):
    """
    2D pooling layer

    Performs 2D mean or max-pooling over the two trailing axes
    of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    See Also
    --------
    MaxPool2DLayer : Shortcut for max pooling layer.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2DLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        self.pad = as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = pool_2d(input,
                         ws=self.pool_size,
                         stride=self.stride,
                         ignore_border=self.ignore_border,
                         pad=self.pad,
                         mode=self.mode,
                         )
        return pooled


class MaxPool1DLayer(Pool1DLayer):
    """
    1D max-pooling layer

    Performs 1D max-pooling over the trailing axis of a 3D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region. If an iterable, it should have a
        single element.

    stride : integer, iterable or ``None``
        The stride between sucessive pooling regions.
        If ``None`` then ``stride == pool_size``.

    pad : integer or iterable
        The number of elements to be added to the input on each side.
        Must be less than stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=0,
                 ignore_border=True, **kwargs):
        super(MaxPool1DLayer, self).__init__(incoming,
                                             pool_size,
                                             stride,
                                             pad,
                                             ignore_border,
                                             mode='max',
                                             **kwargs)


class MaxPool2DLayer(Pool2DLayer):
    """
    2D max-pooling layer

    Performs 2D max-pooling over the two trailing axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, **kwargs):
        super(MaxPool2DLayer, self).__init__(incoming,
                                             pool_size,
                                             stride,
                                             pad,
                                             ignore_border,
                                             mode='max',
                                             **kwargs)

# TODO: add reshape-based implementation to MaxPool*DLayer
# TODO: add MaxPool3DLayer


class Upscale1DLayer(Layer):
    """
    1D upscaling layer

    Performs 1D upscaling over the trailing axis of a 3D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer or iterable
        The scale factor. If an iterable, it should have one element.

    mode : {'repeat', 'dilate'}
        Upscaling mode: repeat element values or upscale leaving zeroes between
        upscaled elements. Default is 'repeat'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, mode='repeat', **kwargs):
        super(Upscale1DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 1)

        if self.scale_factor[0] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

        if mode not in {'repeat', 'dilate'}:
            msg = "Mode must be either 'repeat' or 'dilate', not {0}"
            raise ValueError(msg.format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, = self.scale_factor
        upscaled = input
        if self.mode == 'repeat':
            if a > 1:
                upscaled = T.extra_ops.repeat(upscaled, a, 2)
        elif self.mode == 'dilate':
            if a > 1:
                output_shape = self.get_output_shape_for(input.shape)
                upscaled = T.zeros(shape=output_shape, dtype=input.dtype)
                upscaled = T.set_subtensor(upscaled[:, :, ::a], input)
        return upscaled


class Upscale2DLayer(Layer):
    """
    2D upscaling layer

    Performs 2D upscaling over the two trailing axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.

    mode : {'repeat', 'dilate'}
        Upscaling mode: repeat element values or upscale leaving zeroes between
        upscaled elements. Default is 'repeat'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    Using ``mode='dilate'`` followed by a convolution can be
    realized more efficiently with a transposed convolution, see
    :class:`lasagne.layers.TransposedConv2DLayer`.
    """

    def __init__(self, incoming, scale_factor, mode='repeat', **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

        if mode not in {'repeat', 'dilate'}:
            msg = "Mode must be either 'repeat' or 'dilate', not {0}"
            raise ValueError(msg.format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b = self.scale_factor
        upscaled = input
        if self.mode == 'repeat':
            if b > 1:
                upscaled = T.extra_ops.repeat(upscaled, b, 3)
            if a > 1:
                upscaled = T.extra_ops.repeat(upscaled, a, 2)
        elif self.mode == 'dilate':
            if b > 1 or a > 1:
                output_shape = self.get_output_shape_for(input.shape)
                upscaled = T.zeros(shape=output_shape, dtype=input.dtype)
                upscaled = T.set_subtensor(upscaled[:, :, ::a, ::b], input)
        return upscaled


class Upscale3DLayer(Layer):
    """
    3D upscaling layer

    Performs 3D upscaling over the three trailing axes of a 5D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a cubic scale factor region. If an iterable, it should have three
        elements.

    mode : {'repeat', 'dilate'}
        Upscaling mode: repeat element values or upscale leaving zeroes between
        upscaled elements. Default is 'repeat'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, mode='repeat', **kwargs):
        super(Upscale3DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 3)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1 or \
           self.scale_factor[2] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

        if mode not in {'repeat', 'dilate'}:
            msg = "Mode must be either 'repeat' or 'dilate', not {0}"
            raise ValueError(msg.format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        if output_shape[4] is not None:
            output_shape[4] *= self.scale_factor[2]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b, c = self.scale_factor
        upscaled = input
        if self.mode == 'repeat':
            if c > 1:
                upscaled = T.extra_ops.repeat(upscaled, c, 4)
            if b > 1:
                upscaled = T.extra_ops.repeat(upscaled, b, 3)
            if a > 1:
                upscaled = T.extra_ops.repeat(upscaled, a, 2)
        elif self.mode == 'dilate':
            if c > 1 or b > 1 or a > 1:
                output_shape = self.get_output_shape_for(input.shape)
                upscaled = T.zeros(shape=output_shape, dtype=input.dtype)
                upscaled = T.set_subtensor(
                    upscaled[:, :, ::a, ::b, ::c], input)
        return upscaled


class FeaturePoolLayer(Layer):
    """
    lasagne.layers.FeaturePoolLayer(incoming, pool_size, axis=1,
    pool_function=theano.tensor.max, **kwargs)

    Feature pooling layer

    This layer pools across a given axis of the input. By default this is axis
    1, which corresponds to the feature axis for :class:`DenseLayer`,
    :class:`Conv1DLayer` and :class:`Conv2DLayer`. The layer can be used to
    implement maxout.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer
        the size of the pooling regions, i.e. the number of features / feature
        maps to be pooled together.

    axis : integer
        the axis along which to pool. The default value of ``1`` works
        for :class:`DenseLayer`, :class:`Conv1DLayer` and :class:`Conv2DLayer`.

    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.max`
        (i.e. max-pooling) and can be replaced by any other aggregation
        function.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer requires that the size of the axis along which it pools is a
    multiple of the pool size.
    """

    def __init__(self, incoming, pool_size, axis=1, pool_function=T.max,
                 **kwargs):
        super(FeaturePoolLayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size
        self.axis = axis
        self.pool_function = pool_function

        num_feature_maps = self.input_shape[self.axis]
        if num_feature_maps % self.pool_size != 0:
            raise ValueError("Number of input feature maps (%d) is not a "
                             "multiple of the pool size (pool_size=%d)" %
                             (num_feature_maps, self.pool_size))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # make a mutable copy
        output_shape[self.axis] = input_shape[self.axis] // self.pool_size
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_shape = tuple(input.shape)
        num_feature_maps = input_shape[self.axis]
        num_feature_maps_out = num_feature_maps // self.pool_size

        pool_shape = (input_shape[:self.axis] +
                      (num_feature_maps_out, self.pool_size) +
                      input_shape[self.axis+1:])

        input_reshaped = input.reshape(pool_shape)
        return self.pool_function(input_reshaped, axis=self.axis + 1)


class FeatureWTALayer(Layer):
    """
    'Winner Take All' layer

    This layer performs 'Winner Take All' (WTA) across feature maps: zero out
    all but the maximal activation value within a region.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer
        the number of feature maps per region.

    axis : integer
        the axis along which the regions are formed.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer requires that the size of the axis along which it groups units
    is a multiple of the pool size.
    """

    def __init__(self, incoming, pool_size, axis=1, **kwargs):
        super(FeatureWTALayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size
        self.axis = axis

        num_feature_maps = self.input_shape[self.axis]
        if num_feature_maps % self.pool_size != 0:
            raise ValueError("Number of input feature maps (%d) is not a "
                             "multiple of the region size (pool_size=%d)" %
                             (num_feature_maps, self.pool_size))

    def get_output_for(self, input, **kwargs):
        num_feature_maps = input.shape[self.axis]
        num_pools = num_feature_maps // self.pool_size

        pool_shape = ()
        arange_shuffle_pattern = ()
        for k in range(self.axis):
            pool_shape += (input.shape[k],)
            arange_shuffle_pattern += ('x',)

        pool_shape += (num_pools, self.pool_size)
        arange_shuffle_pattern += ('x', 0)

        for k in range(self.axis + 1, input.ndim):
            pool_shape += (input.shape[k],)
            arange_shuffle_pattern += ('x',)

        input_reshaped = input.reshape(pool_shape)
        max_indices = T.argmax(input_reshaped, axis=self.axis + 1,
                               keepdims=True)

        arange = T.arange(self.pool_size).dimshuffle(*arange_shuffle_pattern)
        mask = T.eq(max_indices, arange).reshape(input.shape)

        return input * mask


class GlobalPoolLayer(Layer):
    """
    lasagne.layers.GlobalPoolLayer(incoming,
    pool_function=theano.tensor.mean, **kwargs)

    Global pooling layer

    This layer pools globally across all trailing dimensions beyond the 2nd.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, pool_function=T.mean, **kwargs):
        super(GlobalPoolLayer, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return input_shape[:2]

    def get_output_for(self, input, **kwargs):
        return self.pool_function(input.flatten(3), axis=2)


def pool_2d_nxn_regions(inputs, output_size, mode='max'):
    """
    Performs a pooling operation that results in a fixed size:
    output_size x output_size.
    Used by SpatialPyramidPoolingLayer. Refer to appendix A in [1]

    Parameters
    ----------
    inputs : a tensor with 4 dimensions (N x C x H x W)
    output_size: integer
        The output size of the pooling operation
    mode : string
        Pooling mode, one of 'max', 'average_inc_pad', 'average_exc_pad'
        Defaults to 'max'.

    Returns a list of tensors, for each output bin.
       The list contains output_size*output_size elements, where
       each element is a 3D tensor (N x C x 1)

    References
    ----------
    .. [1] He, Kaiming et al (2015):
           Spatial Pyramid Pooling in Deep Convolutional Networks
           for Visual Recognition.
           http://arxiv.org/pdf/1406.4729.pdf.
    """

    if mode == 'max':
        pooling_op = T.max
    elif mode in ['average_inc_pad', 'average_exc_pad']:
        pooling_op = T.mean
    else:
        msg = "Mode must be either 'max', 'average_inc_pad' or "
        msg += "'average_exc_pad'. Got '{0}'"
        raise ValueError(msg.format(mode))

    h, w = inputs.shape[2:]

    result = []
    n = float(output_size)

    for row in range(output_size):
        for col in range(output_size):
            start_h = T.floor(row / n * h).astype('int32')
            end_h = T.ceil((row + 1) / n * h).astype('int32')
            start_w = T.floor(col / n * w).astype('int32')
            end_w = T.ceil((col + 1) / n * w).astype('int32')

            pooling_region = inputs[:, :, start_h:end_h, start_w:end_w]
            this_result = pooling_op(pooling_region, axis=(2, 3))
            result.append(this_result.dimshuffle(0, 1, 'x'))
    return result


class SpatialPyramidPoolingLayer(Layer):
    """
    Spatial Pyramid Pooling Layer

    Performs spatial pyramid pooling (SPP) over the input.
    It will turn a 2D input of arbitrary size into an output of fixed
    dimension.
    Hence, the convolutional part of a DNN can be connected to a dense part
    with a fixed number of nodes even if the dimensions of the
    input image are unknown.

    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features.
    :math:`M_i` is given by :math:`n_i * n_i`,
    with :math:`n_i` as the number of pooling operation per dimension in
    level :math:`i`, and we use a list of the :math:`n_i`'s as a
    parameter for SPP-Layer.
    The length of this list is the level of the spatial pyramid.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_dims : list of integers
        The list of :math:`n_i`'s that define the output dimension of each
        pooling level :math:`i`. The length of pool_dims is the level of
        the spatial pyramid.

    mode : string
        Pooling mode, one of 'max', 'average_inc_pad', 'average_exc_pad'
        Defaults to 'max'.

    implementation : string
        Either 'fast' or 'kaiming'. The 'fast' version uses theano's pool_2d
        operation, which is fast but does not work for all input sizes.
        The 'kaiming' mode is slower but implements the pooling as described
        in [1], and works with any input size.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between the convolutional part of a
    DNN and its dense part. Convolutions can be used for
    arbitrary input dimensions, but the size of their output will
    depend on their input dimensions. Connecting the output of the
    convolutional to the dense part then usually demands us to fix
    the dimensions of the network's InputLayer.
    The spatial pyramid pooling layer, however, allows us to leave the
    network input dimensions arbitrary. The advantage over a global
    pooling layer is the added robustness against object deformations
    due to the pooling on different scales.

    References
    ----------
    .. [1] He, Kaiming et al (2015):
           Spatial Pyramid Pooling in Deep Convolutional Networks
           for Visual Recognition.
           http://arxiv.org/pdf/1406.4729.pdf.
    """
    def __init__(self, incoming, pool_dims=[4, 2, 1], mode='max',
                 implementation='fast', **kwargs):
            super(SpatialPyramidPoolingLayer, self).__init__(incoming,
                                                             **kwargs)
            if len(self.input_shape) != 4:
                raise ValueError("Tried to create a SPP layer with "
                                 "input shape %r. Expected 4 input dimensions "
                                 "(batchsize, channels, 2 spatial dimensions)."
                                 % (self.input_shape,))

            if implementation != 'kaiming':  # pragma: no cover
                # Check if the running theano version supports symbolic
                # variables as arguments for pool_2d. This is required
                # unless using implementation='kaiming'
                try:
                    pool_2d(T.tensor4(),
                            ws=T.ivector(),
                            stride=T.ivector(),
                            ignore_border=True,
                            pad=None)
                except ValueError:
                    raise ImportError("SpatialPyramidPoolingLayer with "
                                      "implementation='%s' requires a newer "
                                      "version of theano. Either update "
                                      "theano, or use implementation="
                                      "'kaiming'" % implementation)

            self.mode = mode
            self.implementation = implementation
            self.pool_dims = pool_dims

    def get_output_for(self, input, **kwargs):
        input_size = tuple(symb if fixed is None else fixed
                           for fixed, symb
                           in zip(self.input_shape[2:], input.shape[2:]))
        pool_list = []
        for pool_dim in self.pool_dims:
            if self.implementation == 'kaiming':
                pool_list += pool_2d_nxn_regions(input,
                                                 pool_dim,
                                                 mode=self.mode)
            else:  # pragma: no cover
                win_size = tuple((i + pool_dim - 1) // pool_dim
                                 for i in input_size)
                str_size = tuple(i // pool_dim for i in input_size)

                pool = pool_2d(input,
                               ws=win_size,
                               stride=str_size,
                               mode=self.mode,
                               pad=None,
                               ignore_border=True)
                pool = pool.flatten(3)
                pool_list.append(pool)

        return T.concatenate(pool_list, axis=2)

    def get_output_shape_for(self, input_shape):
        num_features = sum(p*p for p in self.pool_dims)
        return (input_shape[0], input_shape[1], num_features)
