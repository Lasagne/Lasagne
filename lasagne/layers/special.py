import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities
from ..utils import as_tuple
from .base import Layer, MergeLayer


__all__ = [
    "NonlinearityLayer",
    "BiasLayer",
    "InverseLayer",
    "TransformerLayer",
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

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases. If a shared variable or a numpy array
        is provided, the shape must match the incoming shape, skipping those
        axes the biases are shared over (see below for an example). If set to
        ``None``, the layer will have no biases and pass through its input
        unchanged.
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
    >>> l_u = InverseLayer(l2, l1)  # As Deconv2DLayer
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
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015

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
        return (shape[:2] + tuple(None if s is None else int(s / f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        input, theta = inputs
        return _transform(theta, input, self.downsample_factor)


def _transform(theta, input, downsample_factor):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    floatX = theano.config.floatX
    out_height = T.cast(height.astype(floatX) / downsample_factor[0], 'int64')
    out_width = T.cast(width.astype(floatX) / downsample_factor[1], 'int64')
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

    # scale indices from [-1, 1] to [0, width/height].
    x = (x + 1) / 2 * width_f
    y = (y + 1) / 2 * height_f

    # Clip indices to ensure they are not out of bounds.
    max_x = width_f - 1
    max_y = height_f - 1
    x0 = T.clip(x, 0, max_x)
    x1 = T.clip(x + 1, 0, max_x)
    y0 = T.clip(y, 0, max_y)
    y1 = T.clip(y + 1, 0, max_y)

    # We need floatX for interpolation and int64 for indexing.
    x0_f = T.floor(x0)
    x1_f = T.floor(x1)
    y0_f = T.floor(y0)
    y1_f = T.floor(y1)
    x0 = T.cast(x0, 'int64')
    x1 = T.cast(x1, 'int64')
    y0 = T.cast(y0, 'int64')
    y1 = T.cast(y1, 'int64')

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
