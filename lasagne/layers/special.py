import theano
import theano.tensor as T
import numpy as np

from .. import init
from .. import nonlinearities
from ..utils import as_tuple, floatX
from ..random import get_rng
from .base import Layer, MergeLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "NonlinearityLayer",
    "BiasLayer",
    "ScaleLayer",
    "standardize",
    "ExpressionLayer",
    "InverseLayer",
    "TransformerLayer",
    "TPSTransformerLayer",
    "ParametricRectifierLayer",
    "prelu",
    "RandomizedRectifierLayer",
    "rrelu",
]


class NonlinearityLayer(Layer):
    """
    lasagne.layers.NonlinearityLayer(incoming,
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A layer that just applies a nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    """
    def __init__(self, incoming, nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(NonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(input)


class BiasLayer(Layer):
    """
    lasagne.layers.BiasLayer(incoming, b=lasagne.init.Constant(0),
    shared_axes='auto', **kwargs)

    A layer that just adds a (trainable) bias term.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases and pass through its input
        unchanged. Otherwise, the bias shape must match the incoming shape,
        skipping those axes the biases are shared over (see the example below).
        See :func:`lasagne.utils.create_param` for more information.

    shared_axes : 'auto', int or tuple of int
        The axis or axes to share biases over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share biases over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.

    Notes
    -----
    The bias parameter dimensionality is the input dimensionality minus the
    number of axes the biases are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:

    >>> layer = BiasLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.b.get_value().shape
    (30, 50)
    """
    def __init__(self, incoming, b=init.Constant(0), shared_axes='auto',
                 **kwargs):
        super(BiasLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            # default: share biases over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        if b is None:
            self.b = None
        else:
            # create bias parameter, ignoring all dimensions in shared_axes
            shape = [size for axis, size in enumerate(self.input_shape)
                     if axis not in self.shared_axes]
            if any(size is None for size in shape):
                raise ValueError("BiasLayer needs specified input sizes for "
                                 "all axes that biases are not shared over.")
            self.b = self.add_param(b, shape, 'b', regularizable=False)

    def get_output_for(self, input, **kwargs):
        if self.b is not None:
            bias_axes = iter(range(self.b.ndim))
            pattern = ['x' if input_axis in self.shared_axes
                       else next(bias_axes)
                       for input_axis in range(input.ndim)]
            return input + self.b.dimshuffle(*pattern)
        else:
            return input


class ScaleLayer(Layer):
    """
    lasagne.layers.ScaleLayer(incoming, scales=lasagne.init.Constant(1),
    shared_axes='auto', **kwargs)

    A layer that scales its inputs by learned coefficients.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    scales : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for the scale.  The scale
        shape must match the incoming shape, skipping those axes the scales are
        shared over (see the example below).  See
        :func:`lasagne.utils.create_param` for more information.

    shared_axes : 'auto', int or tuple of int
        The axis or axes to share scales over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share scales over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.

    Notes
    -----
    The scales parameter dimensionality is the input dimensionality minus the
    number of axes the scales are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:

    >>> layer = ScaleLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.scales.get_value().shape
    (30, 50)
    """
    def __init__(self, incoming, scales=init.Constant(1), shared_axes='auto',
                 **kwargs):
        super(ScaleLayer, self).__init__(incoming, **kwargs)

        if shared_axes == 'auto':
            # default: share scales over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes

        # create scales parameter, ignoring all dimensions in shared_axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ScaleLayer needs specified input sizes for "
                             "all axes that scales are not shared over.")
        self.scales = self.add_param(
            scales, shape, 'scales', regularizable=False)

    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.scales.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes) for input_axis in range(input.ndim)]
        return input * self.scales.dimshuffle(*pattern)


def standardize(layer, offset, scale, shared_axes='auto'):
    """
    Convenience function for standardizing inputs by applying a fixed offset
    and scale.  This is usually useful when you want the input to your network
    to, say, have zero mean and unit standard deviation over the feature
    dimensions.  This layer allows you to include the appropriate statistics to
    achieve this normalization as part of your network, and applies them to its
    input.  The statistics are supplied as the `offset` and `scale` parameters,
    which are applied to the input by subtracting `offset` and dividing by
    `scale`, sharing dimensions as specified by the `shared_axes` argument.

    Parameters
    ----------
    layer : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    offset : Theano shared variable or numpy array
        The offset to apply (via subtraction) to the axis/axes being
        standardized.
    scale : Theano shared variable or numpy array
        The scale to apply (via division) to the axis/axes being standardized.
    shared_axes : 'auto', int or tuple of int
        The axis or axes to share the offset and scale over. If ``'auto'`` (the
        default), share over all axes except for the second: this will share
        scales over the minibatch dimension for dense layers, and additionally
        over all spatial dimensions for convolutional layers.

    Examples
    --------
    Assuming your training data exists in a 2D numpy ndarray called
    ``training_data``, you can use this function to scale input features to the
    [0, 1] range based on the training set statistics like so:

    >>> import lasagne
    >>> import numpy as np
    >>> training_data = np.random.standard_normal((100, 20))
    >>> input_shape = (None, training_data.shape[1])
    >>> l_in = lasagne.layers.InputLayer(input_shape)
    >>> offset = training_data.min(axis=0)
    >>> scale = training_data.max(axis=0) - training_data.min(axis=0)
    >>> l_std = standardize(l_in, offset, scale, shared_axes=0)

    Alternatively, to z-score your inputs based on training set statistics, you
    could set ``offset = training_data.mean(axis=0)`` and
    ``scale = training_data.std(axis=0)`` instead.
    """
    # Subtract the offset
    layer = BiasLayer(layer, -offset, shared_axes)
    # Do not optimize the offset parameter
    layer.params[layer.b].remove('trainable')
    # Divide by the scale
    layer = ScaleLayer(layer, floatX(1.)/scale, shared_axes)
    # Do not optimize the scales parameter
    layer.params[layer.scales].remove('trainable')
    return layer


class ExpressionLayer(Layer):
    """
    This layer provides boilerplate for a custom layer that applies a
    simple transformation to the input.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    function : callable
        A function to be applied to the output of the previous layer.

    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.

    Notes
    -----
    An :class:`ExpressionLayer` that does not change the shape of the data
    (i.e., is constructed with the default setting of ``output_shape=None``)
    is functionally equivalent to a :class:`NonlinearityLayer`.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, ExpressionLayer
    >>> l_in = InputLayer((32, 100, 20))
    >>> l1 = ExpressionLayer(l_in, lambda X: X.mean(-1), output_shape='auto')
    >>> l1.output_shape
    (32, 100)
    """
    def __init__(self, incoming, function, output_shape=None, **kwargs):
        super(ExpressionLayer, self).__init__(incoming, **kwargs)

        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.function = function

    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            return input_shape
        elif self._output_shape is 'auto':
            input_shape = (0 if s is None else s for s in input_shape)
            X = theano.tensor.alloc(0, *input_shape)
            output_shape = self.function(X).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, input, **kwargs):
        return self.function(input)


class InverseLayer(MergeLayer):
    """
    The :class:`InverseLayer` class performs inverse operations
    for a single layer of a neural network by applying the
    partial derivative of the layer to be inverted with
    respect to its input: transposed layer
    for a :class:`DenseLayer`, deconvolutional layer for
    :class:`Conv2DLayer`, :class:`Conv1DLayer`; or
    an unpooling layer for :class:`MaxPool2DLayer`.

    It is specially useful for building (convolutional)
    autoencoders with tied parameters.

    Note that if the layer to be inverted contains a nonlinearity
    and/or a bias, the :class:`InverseLayer` will include the derivative
    of that in its computation.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    layer : a :class:`Layer` instance or a tuple
        The layer with respect to which the instance of the
        :class:`InverseLayer` is inverse to.

    Examples
    --------
    >>> import lasagne
    >>> from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer
    >>> from lasagne.layers import InverseLayer
    >>> l_in = InputLayer((100, 3, 28, 28))
    >>> l1 = Conv2DLayer(l_in, num_filters=16, filter_size=5)
    >>> l2 = DenseLayer(l1, num_units=20)
    >>> l_u2 = InverseLayer(l2, l2)  # backprop through l2
    >>> l_u1 = InverseLayer(l_u2, l1)  # backprop through l1
    """
    def __init__(self, incoming, layer, **kwargs):

        super(InverseLayer, self).__init__(
            [incoming, layer, layer.input_layer], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[2]

    def get_output_for(self, inputs, **kwargs):
        input, layer_out, layer_in = inputs
        return theano.grad(None, wrt=layer_in, known_grads={layer_out: input})


class TransformerLayer(MergeLayer):
    """
    Spatial transformer layer

    The layer applies an affine transformation on the input. The affine
    transformation is parameterized with six learned parameters [1]_.
    The output is interpolated with a bilinear transformation.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    localization_network : a :class:`Layer` instance
        The network that calculates the parameters of the affine
        transformation. See the example for how to initialize to the identity
        transform.

    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.

    References
    ----------
    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
            Koray Kavukcuoglu (2015):
            Spatial Transformer Networks. NIPS 2015,
            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf

    Examples
    --------
    Here we set up the layer to initially do the identity transform, similarly
    to [1]_. Note that you will want to use a localization with linear output.
    If the output from the localization networks is [t1, t2, t3, t4, t5, t6]
    then t1 and t5 determines zoom, t2 and t4 determines skewness, and t3 and
    t6 move the center position.

    >>> import numpy as np
    >>> import lasagne
    >>> b = np.zeros((2, 3), dtype='float32')
    >>> b[0, 0] = 1
    >>> b[1, 1] = 1
    >>> b = b.flatten()  # identity transform
    >>> W = lasagne.init.Constant(0.0)
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=6, W=W, b=b,
    ... nonlinearity=None)
    >>> l_trans = lasagne.layers.TransformerLayer(l_in, l_loc)
    """
    def __init__(self, incoming, localization_network, downsample_factor=1,
                 **kwargs):
        super(TransformerLayer, self).__init__(
            [incoming, localization_network], **kwargs)
        self.downsample_factor = as_tuple(downsample_factor, 2)

        input_shp, loc_shp = self.input_shapes

        if loc_shp[-1] != 6 or len(loc_shp) != 2:
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, 6)")
        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        input, theta = inputs
        return _transform_affine(theta, input, self.downsample_factor)


def _transform_affine(theta, input, downsample_factor):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(height // downsample_factor[0], 'int64')
    out_width = T.cast(width // downsample_factor[1], 'int64')
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _interpolate(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


class TPSTransformerLayer(MergeLayer):
    """
    Spatial transformer layer

    The layer applies a thin plate spline transformation [2]_ on the input
    as in [1]_. The thin plate spline transform is determined based on the
    movement of some number of control points. The starting positions for
    these control points are fixed. The output is interpolated with a
    bilinear transformation.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    localization_network : a :class:`Layer` instance
        The network that calculates the parameters of the thin plate spline
        transformation as the x and y coordinates of the destination offsets of
        each control point. The output of the localization network  should
        be a 2D tensor, with shape ``(batch_size, 2 * num_control_points)``

    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.

    control_points : integer
        The number of control points to be used for the thin plate spline
        transformation. These points will be arranged as a grid along the
        image, so the value must be a perfect square. Default is 16.

    precompute_grid : 'auto' or boolean
        Flag to precompute the U function [2]_ for the grid and source
        points. If 'auto', will be set to true as long as the input height
        and width are specified. If true, the U function is computed when the
        layer is constructed for a fixed input shape. If false, grid will be
        computed as part of the Theano computational graph, which is
        substantially slower as this computation scales with
        num_pixels*num_control_points. Default is 'auto'.

    References
    ----------
    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
            Koray Kavukcuoglu (2015):
            Spatial Transformer Networks. NIPS 2015,
            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
    .. [2]  Fred L. Bookstein (1989):
            Principal warps: thin-plate splines and the decomposition of
            deformations. IEEE Transactions on
            Pattern Analysis and Machine Intelligence.
            http://doi.org/10.1109/34.24792

    Examples
    --------
    Here, we'll implement an identity transform using a thin plate spline
    transform. First we'll create the destination control point offsets. To
    make everything invariant to the shape of the image, the x and y range
    of the image is normalized to [-1, 1] as in ref [1]_. To replicate an
    identity transform, we'll set the bias to have all offsets be 0. More
    complicated transformations can easily be implemented using different x
    and y offsets (importantly, each control point can have it's own pair of
    offsets).

    >>> import numpy as np
    >>> import lasagne
    >>>
    >>> # Create the network
    >>> # we'll initialize the weights and biases to zero, so it starts
    >>> # as the identity transform (all control point offsets are zero)
    >>> W = b = lasagne.init.Constant(0.0)
    >>>
    >>> # Set the number of points
    >>> num_points = 16
    >>>
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=2*num_points,
    ...                                   W=W, b=b, nonlinearity=None)
    >>> l_trans = lasagne.layers.TPSTransformerLayer(l_in, l_loc,
    ...                                          control_points=num_points)
    """

    def __init__(self, incoming, localization_network, downsample_factor=1,
                 control_points=16, precompute_grid='auto', **kwargs):
        super(TPSTransformerLayer, self).__init__(
                [incoming, localization_network], **kwargs)

        self.downsample_factor = as_tuple(downsample_factor, 2)
        self.control_points = control_points

        input_shp, loc_shp = self.input_shapes

        # Error checking
        if loc_shp[-1] != 2 * control_points or len(loc_shp) != 2:
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, "
                             "2*control_points)")

        if round(np.sqrt(control_points)) != np.sqrt(
                control_points):
            raise ValueError("The number of control points must be"
                             " a perfect square.")

        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

        # Process precompute grid
        can_precompute_grid = all(s is not None for s in input_shp[2:])
        if precompute_grid == 'auto':
            precompute_grid = can_precompute_grid
        elif precompute_grid and not can_precompute_grid:
            raise ValueError("Grid can only be precomputed if the input "
                             "height and width are pre-specified.")
        self.precompute_grid = precompute_grid

        # Create source points and L matrix
        self.right_mat, self.L_inv, self.source_points, self.out_height, \
            self.out_width = _initialize_tps(
                control_points, input_shp, self.downsample_factor,
                precompute_grid)

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        # Get input and destination control points
        input, dest_offsets = inputs
        return _transform_thin_plate_spline(
                dest_offsets, input, self.right_mat, self.L_inv,
                self.source_points, self.out_height, self.out_width,
                self.precompute_grid, self.downsample_factor)


def _transform_thin_plate_spline(
        dest_offsets, input, right_mat, L_inv, source_points, out_height,
        out_width, precompute_grid, downsample_factor):

    num_batch, num_channels, height, width = input.shape
    num_control_points = source_points.shape[1]

    # reshape destination offsets to be (num_batch, 2, num_control_points)
    # and add to source_points
    dest_points = source_points + T.reshape(
            dest_offsets, (num_batch, 2, num_control_points))

    # Solve as in ref [2]
    coefficients = T.dot(dest_points, L_inv[:, 3:].T)

    if precompute_grid:

        # Transform each point on the source grid (image_size x image_size)
        right_mat = T.tile(right_mat.dimshuffle('x', 0, 1), (num_batch, 1, 1))
        transformed_points = T.batched_dot(coefficients, right_mat)

    else:

        # Transformed grid
        out_height = T.cast(height // downsample_factor[0], 'int64')
        out_width = T.cast(width // downsample_factor[1], 'int64')
        orig_grid = _meshgrid(out_height, out_width)
        orig_grid = orig_grid[0:2, :]
        orig_grid = T.tile(orig_grid, (num_batch, 1, 1))

        # Transform each point on the source grid (image_size x image_size)
        transformed_points = _get_transformed_points_tps(
                orig_grid, source_points, coefficients, num_control_points,
                num_batch)

    # Get out new points
    x_transformed = transformed_points[:, 0].flatten()
    y_transformed = transformed_points[:, 1].flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
            input_dim, x_transformed, y_transformed,
            out_height, out_width)

    output = T.reshape(input_transformed,
                       (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _get_transformed_points_tps(new_points, source_points, coefficients,
                                num_points, batch_size):
    """
    Calculates the transformed points' value using the provided coefficients

    :param new_points: num_batch x 2 x num_to_transform tensor
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (num_batch, 2,
        control_points + 3))
    :param num_points: the number of points

    :return: the x and y coordinates of each transformed point. Shape (
        num_batch, 2, num_to_transform)
    """

    # Calculate the U function for the new point and each source point as in
    # ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance

    # Calculate the squared dist between the new point and the source points
    to_transform = new_points.dimshuffle(0, 'x', 1, 2)
    stacked_transform = T.tile(to_transform, (1, num_points, 1, 1))
    r_2 = T.sum(((stacked_transform - source_points.dimshuffle(
            'x', 1, 0, 'x')) ** 2), axis=2)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = T.log(r_2)
    distances = T.switch(T.isnan(log_r_2), r_2 * log_r_2, 0.)

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = T.concatenate([T.ones((batch_size, 1, new_points.shape[2]),
                                        dtype=theano.config.floatX),
                                 new_points], axis=1)
    right_mat = T.concatenate([upper_array, distances], axis=1)

    # Calculate the new value as the dot product
    new_value = T.batched_dot(coefficients, right_mat)
    return new_value


def _U_func_numpy(x1, y1, x2, y2):
    """
    Function which implements the U function from Bookstein paper
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: value of z
    """

    # Return zero if same point
    if x1 == x2 and y1 == y2:
        return 0.

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * np.log(r_2)


def _initialize_tps(num_control_points, input_shape, downsample_factor,
                    precompute_grid):
    """
    Initializes the thin plate spline calculation by creating the source
    point array and the inverted L matrix used for calculating the
    transformations as in ref [2]_

    :param num_control_points: the number of control points. Must be a
        perfect square. Points will be used to generate an evenly spaced grid.
    :param input_shape: tuple with 4 elements specifying the input shape
    :param downsample_factor: tuple with 2 elements specifying the
        downsample for the height and width, respectively
    :param precompute_grid: boolean specifying whether to precompute the
        grid matrix
    :return:
        right_mat: shape (num_control_points + 3, out_height*out_width) tensor
        L_inv: shape (num_control_points + 3, num_control_points + 3) tensor
        source_points: shape (2, num_control_points) tensor
        out_height: tensor constant specifying the ouptut height
        out_width: tensor constant specifying the output width

    """

    # break out input_shape
    _, _, height, width = input_shape

    # Create source grid
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size))

    # Create 2 x num_points array of source points
    source_points = np.vstack(
            (x_control_source.flatten(), y_control_source.flatten()))

    # Convert to floatX
    source_points = source_points.astype(theano.config.floatX)

    # Get number of equations
    num_equations = num_control_points + 3

    # Initialize L to be num_equations square matrix
    L = np.zeros((num_equations, num_equations), dtype=theano.config.floatX)

    # Create P matrix components
    L[0, 3:num_equations] = 1.
    L[1:3, 3:num_equations] = source_points
    L[3:num_equations, 0] = 1.
    L[3:num_equations, 1:3] = source_points.T

    # Loop through each pair of points and create the K matrix
    for point_1 in range(num_control_points):
        for point_2 in range(point_1, num_control_points):

            L[point_1 + 3, point_2 + 3] = _U_func_numpy(
                    source_points[0, point_1], source_points[1, point_1],
                    source_points[0, point_2], source_points[1, point_2])

            if point_1 != point_2:
                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]

    # Invert
    L_inv = np.linalg.inv(L)

    if precompute_grid:
        # Construct grid
        out_height = np.array(height // downsample_factor[0]).astype('int64')
        out_width = np.array(width // downsample_factor[1]).astype('int64')
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, out_width),
                               np.linspace(-1, 1, out_height))
        ones = np.ones(np.prod(x_t.shape))
        orig_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        orig_grid = orig_grid[0:2, :]
        orig_grid = orig_grid.astype(theano.config.floatX)

        # Construct right mat

        # First Calculate the U function for the new point and each source
        # point as in ref [2]
        # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
        # squared distance
        to_transform = orig_grid[:, :, np.newaxis].transpose(2, 0, 1)
        stacked_transform = np.tile(to_transform, (num_control_points, 1, 1))
        stacked_source_points = \
            source_points[:, :, np.newaxis].transpose(1, 0, 2)
        r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=1)

        # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
        log_r_2 = np.log(r_2)
        log_r_2[np.isinf(log_r_2)] = 0.
        distances = r_2 * log_r_2

        # Add in the coefficients for the affine translation (1, x, and y,
        # corresponding to a_1, a_x, and a_y)
        upper_array = np.ones(shape=(1, orig_grid.shape[1]),
                              dtype=theano.config.floatX)
        upper_array = np.concatenate([upper_array, orig_grid], axis=0)
        right_mat = np.concatenate([upper_array, distances], axis=0)

        # Convert to tensors
        out_height = T.as_tensor_variable(out_height)
        out_width = T.as_tensor_variable(out_width)
        right_mat = T.as_tensor_variable(right_mat)

    else:
        out_height = None
        out_width = None
        right_mat = None

    # Convert to tensors
    L_inv = T.as_tensor_variable(L_inv)
    source_points = T.as_tensor_variable(source_points)

    return right_mat, L_inv, source_points, out_height, out_width


class ParametricRectifierLayer(Layer):
    """
    lasagne.layers.ParametricRectifierLayer(incoming,
    alpha=init.Constant(0.25), shared_axes='auto', **kwargs)

    A layer that applies parametric rectify nonlinearity to its input
    following [1]_.

    Equation for the parametric rectifier linear unit:
    :math:`\\varphi(x) = \\max(x,0) + \\alpha \\min(x,0)`

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    alpha : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the alpha values. The
        shape must match the incoming shape, skipping those axes the alpha
        values are shared over (see the example below).
        See :func:`lasagne.utils.create_param` for more information.

    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the parameters of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the parameter over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, which corresponds to a single scalar parameter.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

     References
    ----------
    .. [1] K He, X Zhang et al. (2015):
       Delving Deep into Rectifiers: Surpassing Human-Level Performance on
       ImageNet Classification,
       http://arxiv.org/abs/1502.01852

    Notes
    -----
    The alpha parameter dimensionality is the input dimensionality minus the
    number of axes it is shared over, which matches the same convention as
    the :class:`BiasLayer`.

    >>> layer = ParametricRectifierLayer((20, 3, 28, 28), shared_axes=(0, 3))
    >>> layer.alpha.get_value().shape
    (3, 28)
    """
    def __init__(self, incoming, alpha=init.Constant(0.25), shared_axes='auto',
                 **kwargs):
        super(ParametricRectifierLayer, self).__init__(incoming, **kwargs)
        if shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(shared_axes, int):
            self.shared_axes = (shared_axes,)
        else:
            self.shared_axes = shared_axes

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ParametricRectifierLayer needs input sizes for "
                             "all axes that alpha's are not shared over.")
        self.alpha = self.add_param(alpha, shape, name="alpha",
                                    regularizable=False)

    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.alpha.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes)
                   for input_axis in range(input.ndim)]
        alpha = self.alpha.dimshuffle(pattern)
        return theano.tensor.nnet.relu(input, alpha)


def prelu(layer, **kwargs):
    """
    Convenience function to apply parametric rectify to a given layer's output.
    Will set the layer's nonlinearity to identity if there is one and will
    apply the parametric rectifier instead.

    Parameters
    ----------
    layer: a :class:`Layer` instance
        The `Layer` instance to apply the parametric rectifier layer to;
        note that it will be irreversibly modified as specified above

    **kwargs
        Any additional keyword arguments are passed to the
        :class:`ParametericRectifierLayer`

    Examples
    --------
    Note that this function modifies an existing layer, like this:

    >>> from lasagne.layers import InputLayer, DenseLayer, prelu
    >>> layer = InputLayer((32, 100))
    >>> layer = DenseLayer(layer, num_units=200)
    >>> layer = prelu(layer)

    In particular, :func:`prelu` can *not* be passed as a nonlinearity.
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    return ParametricRectifierLayer(layer, **kwargs)


class RandomizedRectifierLayer(Layer):
    """
    A layer that applies a randomized leaky rectify nonlinearity to its input.

    The randomized leaky rectifier was first proposed and used in the Kaggle
    NDSB Competition, and later evaluated in [1]_. Compared to the standard
    leaky rectifier :func:`leaky_rectify`, it has a randomly sampled slope
    for negative input during training, and a fixed slope during evaluation.

    Equation for the randomized rectifier linear unit during training:
    :math:`\\varphi(x) = \\max((\\sim U(lower, upper)) \\cdot x, x)`

    During evaluation, the factor is fixed to the arithmetic mean of `lower`
    and `upper`.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    lower : Theano shared variable, expression, or constant
        The lower bound for the randomly chosen slopes.

    upper : Theano shared variable, expression, or constant
        The upper bound for the randomly chosen slopes.

    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the random slopes of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the random slope over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, thus using a single random slope.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

     References
    ----------
    .. [1] Bing Xu, Naiyan Wang et al. (2015):
       Empirical Evaluation of Rectified Activations in Convolutional Network,
       http://arxiv.org/abs/1505.00853
    """
    def __init__(self, incoming, lower=0.3, upper=0.8, shared_axes='auto',
                 **kwargs):
        super(RandomizedRectifierLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.lower = lower
        self.upper = upper

        if not isinstance(lower > upper, theano.Variable) and lower > upper:
            raise ValueError("Upper bound for RandomizedRectifierLayer needs "
                             "to be higher than lower bound.")

        if shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(shared_axes, int):
            self.shared_axes = (shared_axes,)
        else:
            self.shared_axes = shared_axes

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true, the arithmetic mean of lower and upper are used for the
            leaky slope.
        """
        if deterministic or self.upper == self.lower:
            return theano.tensor.nnet.relu(input, (self.upper+self.lower)/2.0)
        else:
            shape = list(self.input_shape)
            if any(s is None for s in shape):
                shape = list(input.shape)
            for ax in self.shared_axes:
                shape[ax] = 1

            rnd = self._srng.uniform(tuple(shape),
                                     low=self.lower,
                                     high=self.upper,
                                     dtype=theano.config.floatX)
            rnd = theano.tensor.addbroadcast(rnd, *self.shared_axes)
            return theano.tensor.nnet.relu(input, rnd)


def rrelu(layer, **kwargs):
    """
    Convenience function to apply randomized rectify to a given layer's output.
    Will set the layer's nonlinearity to identity if there is one and will
    apply the randomized rectifier instead.

    Parameters
    ----------
    layer: a :class:`Layer` instance
        The `Layer` instance to apply the randomized rectifier layer to;
        note that it will be irreversibly modified as specified above

    **kwargs
        Any additional keyword arguments are passed to the
        :class:`RandomizedRectifierLayer`

    Examples
    --------
    Note that this function modifies an existing layer, like this:

    >>> from lasagne.layers import InputLayer, DenseLayer, rrelu
    >>> layer = InputLayer((32, 100))
    >>> layer = DenseLayer(layer, num_units=200)
    >>> layer = rrelu(layer)

    In particular, :func:`rrelu` can *not* be passed as a nonlinearity.
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    return RandomizedRectifierLayer(layer, **kwargs)
