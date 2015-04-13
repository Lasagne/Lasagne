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


def pool_output_length(input_length, ds, st, ignore_border=True, pad=0):
    """
    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    ds : integer
        The length of the pooling region
    st : integer
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
    if input_length is None or ds is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - ds + 1
        output_length = (output_length + st - 1) // st

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if st >= ds:
            output_length = (input_length + st - 1) // st
        else:
            output_length = max(
                0, (input_length - ds + st - 1) // st) + 1

    return output_length


class MaxPool1DLayer(Layer):
    """
    This layer performs max pooling over the final dimension
    of a 3D tensor.
    """
    def __init__(self, incoming, ds, st=None, pad=0, ignore_border=False,
                 **kwargs):
        """
        Instantiates the layer.

        Parameters
        ----------
        incoming : a :class:`Layer` instance or tuple
            The layer feeding into this layer, or the expected input shape.
        ds : integer
            The length of the pooling region
        st : integer or None
            The stride between sucessive pooling regions.
            If None, st = ds.
        pad : integer
            The number of elements to be added to the input on each side.
            Must be less than st.
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
        self.ds = ds  # an integer
        self.st = ds if st is None else st
        self.pad = pad
        self.ignore_border = ignore_border

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[-1] = pool_output_length(input_shape[-1],
                                              ds=self.ds,
                                              st=self.st,
                                              ignore_border=self.ignore_border,
                                              pad=self.pad,
                                              )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_4d = T.shape_padright(input, 1)

        pooled = downsample.max_pool_2d(input_4d,
                                        ds=(self.ds, 1),
                                        st=(self.st, 1),
                                        ignore_border=self.ignore_border,
                                        padding=(self.pad, 0),
                                        )
        return pooled[:, :, :, 0]


class MaxPool2DLayer(Layer):
    """
    This layer performs max pooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, st=None,
                 ignore_border=False, pad=(0, 0), **kwargs):
        """
        Instantiates the layer.

        Parameters
        ----------
        incoming : a :class:`Layer` instance or tuple
            The layer feeding into this layer, or the expected input shape.
        ds : integer or iterable
            The length of the pooling region in each dimension
        st : integer, iterable or None
            The strides between sucessive pooling regions in each dimension.
            If None, st = ds.
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

        if (isinstance(ds, int)):
            self.ds = (ds, ds)
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            self.ds = ds

        if st is None:
            self.st = self.ds
        else:
            if (isinstance(st, int)):
                self.st = (st, st)
            else:
                st = tuple(st)
                if len(st) != 2:
                    raise ValueError('st must have len == 2')
                self.st = st

        if (isinstance(pad, int)):
            self.pad = (pad, pad)
        else:
            pad = tuple(pad)
            if len(pad) != 2:
                raise ValueError('pad must have len == 2')
            self.pad = pad

        self.ignore_border = ignore_border

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             ds=self.ds[0],
                                             st=self.st[0],
                                             ignore_border=self.ignore_border,
                                             pad=self.pad[0],
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             ds=self.ds[1],
                                             st=self.st[1],
                                             ignore_border=self.ignore_border,
                                             pad=self.pad[1],
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = downsample.max_pool_2d(input,
                                        ds=self.ds,
                                        st=self.st,
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
