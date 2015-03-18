import theano.tensor as T

from .. import init
from .. import nonlinearities
from ..theano_extensions import conv

from .base import Layer


__all__ = [
    "Conv1DLayer",
    "Conv2DLayer",
]


def conv_output_length(input_length, filter_length,
                       stride, border_mode, pad=0):
    if input_length is None:
        return None
    if border_mode == 'valid':
        output_length = input_length - filter_length + 1
    elif border_mode == 'full':
        output_length = input_length + filter_length - 1
    elif border_mode == 'same':
        output_length = input_length
    elif border_mode == 'pad':
        output_length = input_length + 2 * pad - filter_length + 1
    else:
        raise RuntimeError('Invalid border mode: {0}'.format(border_mode))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


class Conv1DLayer(Layer):
    def __init__(self, incoming, num_filters, filter_length, stride=1,
                 border_mode="valid", untie_biases=False, W=init.Uniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 convolution=conv.conv1d_mc0, **kwargs):
        super(Conv1DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_length = filter_length
        self.stride = stride
        self.border_mode = border_mode
        self.untie_biases = untie_biases
        self.convolution = convolution

        self.W = self.create_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        elif self.untie_biases:
            output_shape = self.get_output_shape()
            self.b = self.create_param(b, (num_filters, output_shape[2]),
                                       name="b")
        else:
            self.b = self.create_param(b, (num_filters,), name="b")

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_length)

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        output_length = conv_output_length(input_shape[2],
                                           self.filter_length,
                                           self.stride,
                                           self.border_mode)

        return (input_shape[0], self.num_filters, output_length)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            conved = self.convolution(input, self.W, subsample=(self.stride,),
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=self.border_mode)
        elif self.border_mode == 'same':
            if self.stride != 1:
                raise NotImplementedError("Strided convolution with "
                                          "border_mode 'same' is not "
                                          "supported by this layer yet.")

            conved = self.convolution(input, self.W, subsample=(self.stride,),
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode='full')
            shift = (self.filter_length - 1) // 2
            conved = conved[:, :, shift:input.shape[2] + shift]
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x')

        return self.nonlinearity(activation)


class Conv2DLayer(Layer):
    def __init__(self, incoming, num_filters, filter_size, strides=(1, 1),
                 border_mode="valid", untie_biases=False, W=init.Uniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.border_mode = border_mode
        self.untie_biases = untie_biases
        self.convolution = convolution

        self.W = self.create_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        elif self.untie_biases:
            output_shape = self.get_output_shape()
            self.b = self.create_param(b, (num_filters, output_shape[2],
                                           output_shape[3]), name="b")
        else:
            self.b = self.create_param(b, (num_filters,), name="b")

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.strides[0],
                                         self.border_mode)

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.strides[1],
                                            self.border_mode)

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            conved = self.convolution(input, self.W, subsample=self.strides,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=self.border_mode)
        elif self.border_mode == 'same':
            if self.strides != (1, 1):
                raise NotImplementedError("Strided convolution with "
                                          "border_mode 'same' is not "
                                          "supported by this layer yet.")

            conved = self.convolution(input, self.W, subsample=self.strides,
                                      image_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

# TODO: add Conv3DLayer
