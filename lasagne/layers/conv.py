import theano.tensor as T

from .. import init
from .. import nonlinearities
from ..utils import as_tuple
from ..theano_extensions import conv

from .base import Layer


__all__ = [
    "Conv1DLayer",
    "Conv2DLayer",
]


def conv_output_length(input_length, filter_size,
                       stride, border_mode, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    border_mode : str, 'valid', 'full', 'same' or 'pad'
        A string indicating the convolution border mode.

        If 'valid', it is assumed that the convolution is only computed where
        the input and the filter fully overlap.

        If 'full', it is assumed that the convolution is computed wherever the
        input and the filter overlap by at least one position.

        If 'same', it is assumed that the convolution is computed wherever the
        input and the filter overlap by at least half the filter size, when the
        filter size is odd. In practice, the input is zero-padded with half the
        filter size at the beginning and half at the end (or one less than half
        in the case of an even filter size). This results in an output length
        that is the same as the input length (for both odd and even filter
        sizes).

        If 'pad', zero padding of `pad` positions is assumed to be applied to
        the input, and then a valid convolution is applied.

    pad : int, optional (default 0)
        If `border_mode` is set to 'pad', this is the size of the padding that
        is applied on both sides of the input. Otherwise, this is ignored.

    Returns
    -------
    int
        The output size corresponding to the given convolution parameters.

    Raises
    ------
    RuntimeError
        When an invalid border_mode string is specified, a `RuntimeError` is
        raised.
    """
    if input_length is None:
        return None
    if border_mode == 'valid':
        output_length = input_length - filter_size + 1
    elif border_mode == 'full':
        output_length = input_length + filter_size - 1
    elif border_mode == 'same':
        output_length = input_length
    elif border_mode == 'pad':
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid border mode: {0}'.format(border_mode))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


class Conv1DLayer(Layer):
    """
    lasagne.layers.Conv1DLayer(incoming, num_filters, filter_size, stride=1,
    border_mode="valid", untie_biases=False, W=lasagne.init.GlorotUniform(),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
    convolution=lasagne.theano_extensions.conv.conv1d_mc0, **kwargs)

    1D convolutional layer

    Performs a 1D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 3D tensor, with shape
        ``(batch_size, num_input_channels, input_length)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable
        An integer or a 1-element tuple specifying the size of the filters.

    stride : int or iterable
        An integer or a 1-element tuple specifying the stride of the
        convolution operation.

    border_mode : str, one of 'valid', 'full', 'same'
        A string indicating the convolution border mode.

        If 'valid', the convolution is only computed where the input and the
        filter fully overlap.

        If 'full', the convolution is computed wherever the input and the
        filter overlap by at least one position.

        If 'same', the convolution is computed wherever the input and the
        filter overlap by at least half the filter size, when the filter size
        is odd. In practice, the input is zero-padded with half the filter size
        at the beginning and half at the end (or one less than half in the case
        of an even filter size). This results in an output length that is the
        same as the input length (for both odd and even filter sizes).

    untie_biases : bool, default False
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        matrix (2D).

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. This should initialize the
        layer weights to a 3D array with shape
        ``(num_filters, num_input_channels, filter_length)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If None is provided, the
        layer will have no biases. This should initialize the layer biases to
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, input_length)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    convolution : callable
        The convolution implementation to use. The
        `lasagne.theano_extensions.conv` module provides some alternative
        implementations for 1D convolutions, because the Theano API only
        features a 2D convolution implementation. Usually it should be fine
        to leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable
        Variable representing the filter weights.

    b : Theano shared variable
        Variable representing the biases.

    Notes
    -----
    Theano's default convolution function (`theano.tensor.nnet.conv.conv2d`)
    does not support the 'same' border mode by default. This layer emulates
    it by performing a 'full' convolution and then cropping the result, which
    may negatively affect performance.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=1,
                 border_mode="valid", untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=conv.conv1d_mc0, **kwargs):
        super(Conv1DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 1)
        self.stride = as_tuple(stride, 1)
        self.border_mode = border_mode
        self.untie_biases = untie_biases
        self.convolution = convolution

        if self.border_mode not in ['valid', 'full', 'same']:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0])

    def get_output_shape_for(self, input_shape):
        output_length = conv_output_length(input_shape[2],
                                           self.filter_size[0],
                                           self.stride[0],
                                           self.border_mode)

        return (input_shape[0], self.num_filters, output_length)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=self.border_mode)
        elif self.border_mode == 'same':
            if self.stride[0] != 1:
                raise NotImplementedError("Strided convolution with "
                                          "border_mode 'same' is not "
                                          "supported by this layer yet.")

            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode='full')
            shift = (self.filter_size[0] - 1) // 2
            conved = conved[:, :, shift:input.shape[2] + shift]

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x')

        return self.nonlinearity(activation)


class Conv2DLayer(Layer):
    """
    lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), border_mode="valid", untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify,
    convolution=theano.tensor.nnet.conv2d, **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    border_mode : str, one of 'valid', 'full', 'same'
        A string indicating the convolution border mode.

        If 'valid', the convolution is only computed where the input and the
        filter fully overlap.

        If 'full', the convolution is computed wherever the input and the
        filter overlap by at least one position.

        If 'same', the convolution is computed wherever the input and the
        filter overlap by at least half the filter size, when the filter size
        is odd. In practice, the input is zero-padded with half the filter size
        at the beginning and half at the end (or one less than half in the case
        of an even filter size). This results in an output length that is the
        same as the input length (for both odd and even filter sizes).

    untie_biases : bool, default False
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. This should initialize the
        layer weights to a 4D array with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If None is provided, the
        layer will have no biases. This should initialize the layer biases to
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, input_rows, input_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable
        Variable representing the filter weights.

    b : Theano shared variable
        Variable representing the biases.

    Notes
    -----
    Theano's default convolution function (`theano.tensor.nnet.conv.conv2d`)
    does not support the 'same' border mode by default. This layer emulates
    it by performing a 'full' convolution and then cropping the result, which
    may negatively affect performance.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 border_mode="valid", untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.border_mode = border_mode
        self.untie_biases = untie_biases
        self.convolution = convolution

        if self.border_mode not in ['valid', 'full', 'same']:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         self.border_mode)

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            self.border_mode)

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=self.border_mode)
        elif self.border_mode == 'same':
            if self.stride != (1, 1):
                raise NotImplementedError("Strided convolution with "
                                          "border_mode 'same' is not "
                                          "supported by this layer yet.")

            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

# TODO: add Conv3DLayer
