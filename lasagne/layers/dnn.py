import theano
from theano.sandbox.cuda import dnn

from .. import init
from .. import nonlinearities
from .base import Layer

from .conv import conv_output_length
from ..utils import as_tuple

if not theano.config.device.startswith("gpu") or not dnn.dnn_available():
    raise ImportError("dnn not available")


__all__ = [
    "Pool2DDNNLayer",
    "MaxPool2DDNNLayer",
    "Conv2DDNNLayer",
]


class DNNLayer(Layer):
    pass


class Pool2DDNNLayer(DNNLayer):

    def __init__(self, incoming, pool_size, stride=None, mode='max', **kwargs):
        if 'pad' in kwargs:
            raise NotImplementedError("Pool2DDNNLayer does not "
                                      "support padding")
        super(Pool2DDNNLayer, self).__init__(incoming, **kwargs)
        self.pool_size = as_tuple(pool_size, 2)
        self.mode = mode
        self.stride = as_tuple(stride, 2) if stride is not None else pool_size

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        output_shape[2] = (
            output_shape[2] - self.pool_size[0]) // self.stride[0] + 1
        output_shape[3] = (
            output_shape[3] - self.pool_size[1]) // self.stride[1] + 1
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return dnn.dnn_pool(input, self.pool_size, self.stride, self.mode)


class MaxPool2DDNNLayer(Pool2DDNNLayer):  # for consistency

    def __init__(self, incoming, pool_size, stride=None, **kwargs):
        super(MaxPool2DDNNLayer, self).__init__(incoming, pool_size, stride,
                                                mode='max', **kwargs)


class Conv2DDNNLayer(DNNLayer):

    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 border_mode=None, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 pad=None, flip_filters=False, **kwargs):
        super(Conv2DDNNLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and "
                               "'pad'. To avoid ambiguity, please specify "
                               "only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0, 0)
            self.border_mode = 'valid'
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0, 0)
                self.border_mode = 'valid'
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] - 1)
                self.border_mode = 'full'
            elif border_mode == 'same':
                # dnn_conv does not support same, so we just specify
                # padding directly.
                # only works for odd filter size, but the even filter size
                # case is probably not worth supporting.
                self.pad = ((self.filter_size[0] - 1) // 2,
                            (self.filter_size[1] - 1) // 2)
                self.border_mode = None
            else:
                raise RuntimeError("Unsupported border_mode for "
                                   "Conv2DDNNLayer: %s" % border_mode)
        else:
            self.pad = as_tuple(pad, 2)
            self.border_mode = None

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
        batch_size = input_shape[0]

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         'pad', self.pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            'pad', self.pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        # if 'border_mode' is one of 'valid' or 'full' use that.
        # else use pad directly.
        border_mode = (self.border_mode if (self.border_mode is not None)
                       else self.pad)

        conved = dnn.dnn_conv(img=input,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=border_mode,
                              conv_mode=conv_mode
                              )

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)
