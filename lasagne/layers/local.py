import theano.tensor as T

from .. import init
from .. import nonlinearities

from .conv import Conv2DLayer


__all__ = [
    "LocallyConnected2DLayer",
]


class LocallyConnected2DLayer(Conv2DLayer):
    """
    lasagne.layers.LocallyConnected2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad='same', untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    channelwise=False, **kwargs)

    2D locally connected layer

    Performs an operation similar to a 2D convolution but without the weight
    sharing, then optionally adds a bias and applies an elementwise
    nonlinearity.

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
        This implementation only supports unit stride, the argument is
        provided for compatibility to convolutional layers only.

    pad : int, iterable of int, or 'valid' (default: 'same')
        The amount of implicit zero padding of the input.
        This implementation only supports 'same' padding, the argument is
        provided for compatibility to other convolutional layers only.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        If ``channelwise`` is set to ``False``, the weights should be a 6D
        tensor with shape ``(num_filters, num_input_channels, filter_rows,
        filter_columns, output_rows, output_columns)``. If ``channelwise`` is
        set to ``True``, the weights should be a 5D tensor with shape
        ``(num_filters, filter_rows, filter_columns, output_rows,
        output_columns)``.
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

    flip_filters : bool (default: True)
        Whether to flip the filters before multiplying them over the input,
        similar to a convolution (this is the default), or not to flip them,
        similar to a correlation.

    channelwise : bool (default: False)
        If ``False``, each filter interacts will all of the input channels as
        in a convolution. If ``True``, each filter only interacts with the
        corresponding input channel. That is, each output channel only depends
        on its filter and on the input channel at the same channel index.
        In this case, the number of output channels (i.e. number of filters)
        should be equal to the number of input channels.

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
    This implementation computes the output tensor by iterating over the filter
    weights and multiplying them with shifted versions of the input tensor.
    This implementation assumes no stride, 'same' padding and no dilation.

    Raises
    ------
    ValueError
        When ``channelwise`` is set to ``True`` and the number of filters
        differs from the number of input channels, a `ValueError` is raised.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad='same', untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 channelwise=False, **kwargs):
        self.channelwise = channelwise
        super(LocallyConnected2DLayer, self).__init__(
            incoming, num_filters, filter_size, stride=stride, pad=pad,
            untie_biases=untie_biases, W=W, b=b, nonlinearity=nonlinearity,
            flip_filters=flip_filters, **kwargs)
        # require no stride
        if self.stride != (1, 1):
            raise NotImplementedError(
                "LocallyConnected2DLayer requires stride=1 / (1, 1), but got "
                "%r." % (stride,))
        # require same convolution
        if self.pad != 'same':
            raise NotImplementedError(
                "LocallyConnected2DLayer requires pad='same', but got %r." %
                (pad,))

    def get_W_shape(self):
        if any(s is None for s in self.input_shape[1:]):
            raise ValueError(
                "A LocallyConnected2DLayer requires a fixed input shape "
                "(except for the batch size). Got %r." % (self.input_shape,))
        num_input_channels = self.input_shape[1]
        output_shape = self.get_output_shape_for(self.input_shape)
        if self.channelwise:
            if self.channelwise and self.num_filters != num_input_channels:
                raise ValueError("num_filters and the number of input "
                                 "channels should match when channelwise is "
                                 "true, but got num_filters=%r and %d input "
                                 "channels" %
                                 (self.num_filters, num_input_channels))
            return (self.num_filters,) + self.filter_size + output_shape[-2:]
        else:
            return (self.num_filters, num_input_channels) + \
                   self.filter_size + output_shape[-2:]

    def convolve(self, input, **kwargs):
        output_shape = self.output_shape

        # start with ii == jj == 0 case to initialize tensor
        i = self.filter_size[0] // 2
        j = self.filter_size[1] // 2
        filter_h_ind = -i-1 if self.flip_filters else i
        filter_w_ind = -j-1 if self.flip_filters else j
        if self.channelwise:
            conved = input * self.W[:, filter_h_ind, filter_w_ind, :, :]
        else:
            conved = \
                (input[:, None, :, :, :] *
                 self.W[:, :, filter_h_ind, filter_w_ind, :, :]).sum(axis=-3)

        for i in range(self.filter_size[0]):
            filter_h_ind = -i-1 if self.flip_filters else i
            ii = i - (self.filter_size[0] // 2)
            input_h_slice = slice(
                max(ii, 0), min(ii + output_shape[-2], output_shape[-2]))
            output_h_slice = slice(
                max(-ii, 0), min(-ii + output_shape[-2], output_shape[-2]))

            for j in range(self.filter_size[1]):
                filter_w_ind = -j-1 if self.flip_filters else j
                jj = j - (self.filter_size[1] // 2)
                input_w_slice = slice(
                    max(jj, 0), min(jj + output_shape[-1], output_shape[-1]))
                output_w_slice = slice(
                    max(-jj, 0), min(-jj + output_shape[-1], output_shape[-1]))
                # skip this case since it was done at the beginning
                if ii == jj == 0:
                    continue
                if self.channelwise:
                    inc = (input[:, :, input_h_slice, input_w_slice] *
                           self.W[:, filter_h_ind, filter_w_ind,
                                  output_h_slice, output_w_slice])
                else:
                    inc = (input[:, None, :, input_h_slice, input_w_slice] *
                           self.W[:, :, filter_h_ind, filter_w_ind,
                                  output_h_slice, output_w_slice]).sum(axis=-3)
                conved = T.inc_subtensor(
                    conved[:, :, output_h_slice, output_w_slice], inc)
        return conved
