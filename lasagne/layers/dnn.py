import theano
from theano.sandbox.cuda import dnn

from .. import init
from .. import nonlinearities
from .base import Layer

from .conv import conv_output_length, BaseConvLayer
from .pool import pool_output_length
from .normalization import BatchNormLayer
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
    "SpatialPyramidPoolingDNNLayer",
    "BatchNormDNNLayer",
    "batch_norm_dnn",
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
        num_input_channels, input_depth, input_rows, input_columns)``.

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
        num_input_channels, filter_depth, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_depth, output_rows, output_columns)`` instead.
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


class SpatialPyramidPoolingDNNLayer(Layer):
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
        Pooling mode, one of 'max', 'average_inc_pad' or 'average_exc_pad'.
        Defaults to 'max'.

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
    def __init__(self, incoming, pool_dims=[4, 2, 1], mode='max', **kwargs):
            super(SpatialPyramidPoolingDNNLayer, self).__init__(incoming,
                                                                **kwargs)
            if len(self.input_shape) != 4:
                raise ValueError("Tried to create a SPP layer with "
                                 "input shape %r. Expected 4 input dimensions "
                                 "(batchsize, channels, 2 spatial dimensions)."
                                 % (self.input_shape,))
            self.mode = mode
            self.pool_dims = pool_dims

    def get_output_for(self, input, **kwargs):
        input_size = tuple(symb if fixed is None else fixed
                           for fixed, symb
                           in zip(self.input_shape[2:], input.shape[2:]))
        pool_list = []
        for pool_dim in self.pool_dims:
            win_size = tuple((i + pool_dim - 1) // pool_dim
                             for i in input_size)
            str_size = tuple(i // pool_dim for i in input_size)

            pool = dnn.dnn_pool(input, win_size, str_size, self.mode, (0, 0))
            pool = pool.flatten(3)
            pool_list.append(pool)

        return theano.tensor.concatenate(pool_list, axis=2)

    def get_output_shape_for(self, input_shape):
        num_features = sum(p*p for p in self.pool_dims)
        return (input_shape[0], input_shape[1], num_features)


class BatchNormDNNLayer(BatchNormLayer):
    """
    lasagne.layers.BatchNormDNNLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    This is a drop-in replacement for :class:`lasagne.layers.BatchNormLayer`
    that uses cuDNN for improved performance and reduced memory usage.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. Only supports ``'auto'``
        and the equivalent axes list, or ``0`` and ``(0,)`` to normalize over
        the minibatch dimension only.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems. Must
        not be smaller than ``1e-5``.
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm_dnn` modifies an existing layer to
    insert cuDNN batch normalization in front of its nonlinearity.

    For further information, see :class:`lasagne.layers.BatchNormLayer`. This
    implementation is fully compatible, except for restrictions on the `axes`
    and `epsilon` arguments.

    See also
    --------
    batch_norm_dnn : Convenience function to apply batch normalization
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormDNNLayer, self).__init__(
                incoming, axes, epsilon, alpha, beta, gamma, mean, inv_std,
                **kwargs)
        all_but_second_axis = (0,) + tuple(range(2, len(self.input_shape)))
        if self.axes not in ((0,), all_but_second_axis):
            raise ValueError("BatchNormDNNLayer only supports normalization "
                             "across the first axis, or across all but the "
                             "second axis, got axes=%r" % (axes,))

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]
        # and prepare the converse pattern removing those broadcastable axes
        unpattern = [d for d in range(input.ndim) if d not in self.axes]

        # call cuDNN if needed, obtaining normalized outputs and statistics
        if not use_averages or update_averages:
            # cuDNN requires beta/gamma tensors; create them if needed
            shape = tuple(s for (d, s) in enumerate(input.shape)
                          if d not in self.axes)
            gamma = self.gamma or theano.tensor.ones(shape)
            beta = self.beta or theano.tensor.zeros(shape)
            mode = 'per-activation' if self.axes == (0,) else 'spatial'
            (normalized,
             input_mean,
             input_inv_std) = dnn.dnn_batch_normalization_train(
                    input, gamma.dimshuffle(pattern), beta.dimshuffle(pattern),
                    mode, self.epsilon)

        # normalize with stored averages, if needed
        if use_averages:
            mean = self.mean.dimshuffle(pattern)
            inv_std = self.inv_std.dimshuffle(pattern)
            gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
            beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
            normalized = (input - mean) * (gamma * inv_std) + beta

        # update stored averages, if needed
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean.dimshuffle(unpattern))
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std.dimshuffle(unpattern))
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            dummy = 0 * (running_mean + running_inv_std).dimshuffle(pattern)
            normalized = normalized + dummy

        return normalized


def batch_norm_dnn(layer, **kwargs):
    """
    Apply cuDNN batch normalization to an existing layer. This is a drop-in
    replacement for :func:`lasagne.layers.batch_norm`; see there for further
    information.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        modified as specified in :func:`lasagne.layers.batch_norm`
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`BatchNormDNNLayer` constructor.

    Returns
    -------
    BatchNormDNNLayer or NonlinearityLayer instance
        A batch normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_bn'))
    layer = BatchNormDNNLayer(layer, name=bn_name, **kwargs)
    if nonlinearity is not None:
        from .special import NonlinearityLayer
        nonlin_name = bn_name and bn_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer


if not hasattr(dnn, 'dnn_batch_normalization_train'):
    # Hide cuDNN-based batch normalization for old Theano versions
    del BatchNormDNNLayer, batch_norm_dnn
    __all__.remove('BatchNormDNNLayer')
    __all__.remove('batch_norm_dnn')
