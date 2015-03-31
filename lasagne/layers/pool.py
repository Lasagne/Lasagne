import numpy as np
import theano.tensor as T

from .base import Layer

from theano.tensor.signal import downsample


__all__ = [
    "MaxPool1DLayer",
    "MaxPool2DLayer",
    "FeaturePoolLayer",
    "FeatureWTALayer",
    "GlobalPoolLayer",
]


def pool_output_length(input_length, pool_length, ignore_border=True):
    '''Compute the output length of a pooling operator
    along a particular dimension.

    Parameters
    ----------
    input_length
    pool_length
        Shape of the input and pooling operator in the chosen dimension

    ignore_border:
        if True, the output length is rounded down.
        if False, it is rounded up

    Returns
    -------
    output_length
        * None if either input is None
        * `input_length / pool_length` otherwise
    '''

    if input_length is None or pool_length is None:
        return None

    if ignore_border:
        # Round shape down when we ignore the border
        return input_length // pool_length

    # Otherwise, round the shape up
    return int(np.ceil(float(input_length) / pool_length))


class MaxPool1DLayer(Layer):
    def __init__(self, incoming, ds, ignore_border=False, **kwargs):
        super(MaxPool1DLayer, self).__init__(incoming, **kwargs)
        self.ds = ds  # an integer
        self.ignore_border = ignore_border

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             self.ds,
                                             ignore_border=self.ignore_border)

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_4d = T.shape_padright(input, 1)
        pooled = downsample.max_pool_2d(input_4d, (self.ds, 1),
                                        self.ignore_border)
        return pooled[:, :, :, 0]


class MaxPool2DLayer(Layer):
    def __init__(self, incoming, ds, ignore_border=False, **kwargs):
        super(MaxPool2DLayer, self).__init__(incoming, **kwargs)
        self.ds = ds  # a tuple
        self.ignore_border = ignore_border

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             self.ds[0],
                                             ignore_border=self.ignore_border)

        output_shape[3] = pool_output_length(input_shape[3],
                                             self.ds[1],
                                             ignore_border=self.ignore_border)

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return downsample.max_pool_2d(input, self.ds, self.ignore_border)


# TODO: add reshape-based implementation to MaxPool*DLayer
# TODO: add MaxPool3DLayer


class FeaturePoolLayer(Layer):
    """
    Pooling across feature maps. This can be used to implement maxout.
    IMPORTANT: this layer requires that the number of feature maps is
    a multiple of the pool size.
    """
    def __init__(self, incoming, ds, axis=1, pool_function=T.max, **kwargs):
        """
        ds: the number of feature maps to be pooled together
        axis: the axis along which to pool. The default value of 1 works
        for DenseLayer and Conv*DLayers
        pool_function: the pooling function to use
        """
        super(FeaturePoolLayer, self).__init__(incoming, **kwargs)
        self.ds = ds
        self.axis = axis
        self.pool_function = pool_function

        num_feature_maps = self.input_shape[self.axis]
        if num_feature_maps % self.ds != 0:
            raise RuntimeError("Number of input feature maps (%d) is not a "
                               "multiple of the pool size (ds=%d)" %
                               (num_feature_maps, self.ds))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # make a mutable copy
        output_shape[self.axis] = pool_output_length(input_shape[self.axis],
                                                     self.ds)
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        num_feature_maps = input.shape[self.axis]
        num_feature_maps_out = num_feature_maps // self.ds

        pool_shape = ()
        for k in range(self.axis):
            pool_shape += (input.shape[k],)
        pool_shape += (num_feature_maps_out, self.ds)
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
    def __init__(self, incoming, ds, axis=1, **kwargs):
        """
        ds: the number of feature maps per group. This is called 'ds'
        for consistency with the pooling layers, even though this
        layer does not actually perform a downsampling operation.
        axis: the axis along which the groups are formed.
        """
        super(FeatureWTALayer, self).__init__(incoming, **kwargs)
        self.ds = ds
        self.axis = axis

        num_feature_maps = self.input_shape[self.axis]
        if num_feature_maps % self.ds != 0:
            raise RuntimeError("Number of input feature maps (%d) is not a "
                               "multiple of the group size (ds=%d)" %
                               (num_feature_maps, self.ds))

    def get_output_for(self, input, **kwargs):
        num_feature_maps = input.shape[self.axis]
        num_pools = num_feature_maps // self.ds

        pool_shape = ()
        arange_shuffle_pattern = ()
        for k in range(self.axis):
            pool_shape += (input.shape[k],)
            arange_shuffle_pattern += ('x',)

        pool_shape += (num_pools, self.ds)
        arange_shuffle_pattern += ('x', 0)

        for k in range(self.axis + 1, input.ndim):
            pool_shape += (input.shape[k],)
            arange_shuffle_pattern += ('x',)

        input_reshaped = input.reshape(pool_shape)
        max_indices = T.argmax(input_reshaped, axis=self.axis + 1,
                               keepdims=True)

        arange = T.arange(self.ds).dimshuffle(*arange_shuffle_pattern)
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
