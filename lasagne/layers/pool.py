import theano.tensor as T

from .base import Layer
from ..utils import as_tuple

from theano.tensor.signal import downsample


__all__ = [
    "MaxPool1DLayer",
    "MaxPool2DLayer",
    "FeaturePoolLayer",
    "FeatureWTALayer",
    "GlobalPoolLayer",
]


def pool_output_length(input_length, pool_size, stride,
                       ignore_border=True, pad=0):
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
        If True, partial pooling regions will be ignored.
        Must be True if pad != 0.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When `ignore_border == True`, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If `ignore_border == False`, a single partial pooling region is
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


class MaxPool1DLayer(Layer):

    """
    This layer performs max pooling over the final dimension
    of a 3D tensor.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=0,
                 ignore_border=False, **kwargs):
        """
        Instantiates the layer.

        Parameters
        ----------
        incoming : a :class:`Layer` instance or tuple
            The layer feeding into this layer, or the expected input shape.
        pool_size : integer
            The length of the pooling region
        stride : integer or None
            The stride between sucessive pooling regions.
            If None, stride = pool_size.
        pad : integer
            The number of elements to be added to the input on each side.
            Must be less than stride.
        ignore_border : bool
            If True, partial pooling regions will be ignored.
            Must be True if pad != 0.

        Notes
        -----
        The value used to pad the input is chosen to be less than
        the minimum of the input, so that the output of each pooling region
        always corresponds to some element in the unpadded input region.
        """
        super(MaxPool1DLayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size  # an integer
        self.stride = pool_size if stride is None else stride
        self.pad = pad
        self.ignore_border = ignore_border

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[-1] = pool_output_length(input_shape[-1],
                                              pool_size=self.pool_size,
                                              stride=self.stride,
                                              ignore_border=self.ignore_border,
                                              pad=self.pad,
                                              )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_4d = T.shape_padright(input, 1)

        pooled = downsample.max_pool_2d(input_4d,
                                        ds=(self.pool_size, 1),
                                        st=(self.stride, 1),
                                        ignore_border=self.ignore_border,
                                        padding=(self.pad, 0),
                                        )
        return pooled[:, :, :, 0]


class MaxPool2DLayer(Layer):

    """
    This layer performs max pooling over the last two dimensions
    of a 4D tensor.
    """

    def __init__(self, incoming, pool_size, stride=None,
                 ignore_border=False, pad=(0, 0), **kwargs):
        """
        Instantiates the layer.

        Parameters
        ----------
        incoming : a :class:`Layer` instance or tuple
            The layer feeding into this layer, or the expected input shape.
        pool_size : integer or iterable
            The length of the pooling region in each dimension
        stride : integer, iterable or None
            The strides between sucessive pooling regions in each dimension.
            If None, stride = pool_size.
        pad : integer or iterable
            Number of elements to be added on each side of the input
            in each dimension. Each value must be less than
            the corresponding stride.
        ignore_border : bool
            If True, partial pooling regions will be ignored.
            Must be True if pad != (0, 0).

        Notes
        -----
        The value used to pad the input is chosen to be less than
        the minimum of the input, so that the output of each pooling region
        always corresponds to some element in the unpadded input region.
        """
        super(MaxPool2DLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        self.pad = as_tuple(pad, 2)

        self.ignore_border = ignore_border

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             ignore_border=self.ignore_border,
                                             pad=self.pad[0],
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             ignore_border=self.ignore_border,
                                             pad=self.pad[1],
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = downsample.max_pool_2d(input,
                                        ds=self.pool_size,
                                        st=self.stride,
                                        ignore_border=self.ignore_border,
                                        padding=self.pad,
                                        )
        return pooled


# TODO: add reshape-based implementation to MaxPool*DLayer
# TODO: add MaxPool3DLayer


class FeaturePoolLayer(Layer):

    """
    Pooling across feature maps. This can be used to implement maxout.
    IMPORTANT: this layer requires that the number of feature maps is
    a multiple of the pool size.
    """

    def __init__(self, incoming, pool_size, axis=1, pool_function=T.max,
                 **kwargs):
        """
        Instrideantiates the layer.

        Parameters
        ----------
        pool_size : integer
            the number of feature maps to be pooled together
        axis : integer
            the axis along which to pool. The default value of 1 works
            for DenseLayer and Conv*DLayers
        pool_function : the pooling function to use
        """
        super(FeaturePoolLayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size
        self.axis = axis
        self.pool_function = pool_function

        num_feature_maps = self.input_shape[self.axis]
        if num_feature_maps % self.pool_size != 0:
            raise RuntimeError("Number of input feature maps (%d) is not a "
                               "multiple of the pool size (pool_size=%d)" %
                               (num_feature_maps, self.pool_size))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # make a mutable copy
        output_shape[self.axis] = pool_output_length(input_shape[self.axis],
                                                     self.pool_size)
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        num_feature_maps = input.shape[self.axis]
        num_feature_maps_out = num_feature_maps // self.pool_size

        pool_shape = ()
        for k in range(self.axis):
            pool_shape += (input.shape[k],)
        pool_shape += (num_feature_maps_out, self.pool_size)
        for k in range(self.axis + 1, input.ndim):
            pool_shape += (input.shape[k],)

        input_reshaped = input.reshape(pool_shape)
        return self.pool_function(input_reshaped, axis=self.axis + 1)


class FeatureWTALayer(Layer):

    """
    Perform 'Winner Take All' across feature maps: zero out all but
    the maximal activation value within a group of features.
    IMPORTANT: this layer requires that the number of feature maps is
    a multiple of the pool size.
    """

    def __init__(self, incoming, pool_size, axis=1, **kwargs):
        """
        Instantiates the layer.

        Parameters
        ----------
        pool_size : integer
            the number of feature maps per group.
        axis : integer
            the axis along which the groups are formed.
        """
        super(FeatureWTALayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size
        self.axis = axis

        num_feature_maps = self.input_shape[self.axis]
        if num_feature_maps % self.pool_size != 0:
            raise RuntimeError("Number of input feature maps (%d) is not a "
                               "multiple of the group size (pool_size=%d)" %
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
    Layer that pools globally across all trailing dimensions beyond the 2nd.
    """

    def __init__(self, incoming, pool_function=T.mean, **kwargs):
        super(GlobalPoolLayer, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return input_shape[:2]

    def get_output_for(self, input, **kwargs):
        return self.pool_function(input.flatten(3), axis=2)
