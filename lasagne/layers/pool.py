import theano.tensor as T

from .base import Layer
from ..utils import as_tuple

from theano.tensor.signal import downsample


__all__ = [
    "MaxPool1DLayer",
    "MaxPool2DLayer",
    "Pool1DLayer",
    "Pool2DLayer",
    "Upscale1DLayer",
    "Upscale2DLayer",
    "FeaturePoolLayer",
    "FeatureWTALayer",
    "GlobalPoolLayer",
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

        pooled = downsample.max_pool_2d(input_4d,
                                        ds=(self.pool_size[0], 1),
                                        st=(self.stride[0], 1),
                                        ignore_border=self.ignore_border,
                                        padding=(self.pad[0], 0),
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
        pooled = downsample.max_pool_2d(input,
                                        ds=self.pool_size,
                                        st=self.stride,
                                        ignore_border=self.ignore_border,
                                        padding=self.pad,
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

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, **kwargs):
        super(Upscale1DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 1)

        if self.scale_factor[0] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, = self.scale_factor
        upscaled = input
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, 2)
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

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

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
        if b > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, 3)
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, 2)
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
