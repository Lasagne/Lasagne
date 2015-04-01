"""
GpuCorrMM-based convolutional layers
"""

import theano

from .. import init
from .. import nonlinearities

from .base import Layer

from .conv import conv_output_length

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.blas import GpuCorrMM


__all__ = [
    "MMLayer",
    "Conv2DMMLayer",
]


if not theano.config.device.startswith("gpu"):
    raise ImportError("requires a GPU to work")


# base class for all layers that rely on GpuCorrMM directly
class MMLayer(Layer):
    pass


class Conv2DMMLayer(MMLayer):
    def __init__(self, incoming, num_filters, filter_size, strides=(1, 1),
                 border_mode=None, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 pad=None, flip_filters=False, **kwargs):
        super(Conv2DMMLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and "
                               "'pad'. To avoid ambiguity, please specify "
                               "only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0, 0)
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0, 0)
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] - 1)
            elif border_mode == 'same':
                # only works for odd filter size, but the even filter size case
                # is probably not worth supporting.
                self.pad = ((self.filter_size[0] - 1) // 2,
                            (self.filter_size[1] - 1) // 2)
            else:
                raise RuntimeError("Unsupported border_mode for "
                                   "Conv2DMMLayer: %s" % border_mode)
        else:
            self.pad = pad

        self.W = self.create_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        elif self.untie_biases:
            output_shape = self.get_output_shape()
            self.b = self.create_param(b, (num_filters, output_shape[2],
                                           output_shape[3]), name="b")
        else:
            self.b = self.create_param(b, (num_filters,), name="b")

        self.corr_mm_op = GpuCorrMM(subsample=self.strides, pad=self.pad)

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
                                         self.strides[0],
                                         'pad', self.pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.strides[1],
                                            'pad', self.pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        filters = self.W
        if self.flip_filters:
            filters = filters[:, :, ::-1, ::-1]  # flip top-down, left-right

        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)
        conved = self.corr_mm_op(contiguous_input, contiguous_filters)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)
