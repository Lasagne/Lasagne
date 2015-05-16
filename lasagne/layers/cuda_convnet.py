import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer

from .conv import conv_output_length
from ..utils import as_tuple

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs

__all__ = [
    "CCLayer",
    "Conv2DCCLayer",
    "MaxPool2DCCLayer",
    "ShuffleBC01ToC01BLayer",
    "bc01_to_c01b",
    "ShuffleC01BToBC01Layer",
    "c01b_to_bc01",
    "NINLayer_c01b",
]


if not theano.config.device.startswith("gpu"):
    raise ImportError("requires a GPU to work")


# TODO: make sure to document the limitations and 'best practices'
# (i.e. minibatch size % 128 == 0)
# TODO: see if the 'dimshuffle' logic can be put in the base class instead.


# base class for all layers that use ops from pylearn2.sandbox.cuda_convnet
class CCLayer(Layer):
    pass


class Conv2DCCLayer(CCLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 border_mode=None, untie_biases=False, W=None,
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 pad=None, dimshuffle=True, flip_filters=False, partial_sum=1,
                 **kwargs):
        super(Conv2DCCLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        filter_size = as_tuple(filter_size, 2)
        stride = as_tuple(stride, 2)

        if filter_size[0] != filter_size[1]:
            raise RuntimeError("Conv2DCCLayer only supports square filters, "
                               "but filter_size=(%d, %d)" % filter_size)

        if stride[0] != stride[1]:
            raise RuntimeError("Conv2DCCLayer only supports square strides, "
                               "but stride=(%d, %d)" % stride)

        if num_filters % 16 != 0:
            raise RuntimeError("Conv2DCCLayer requires num_filters to be a "
                               "multiple of 16, but num_filters is "
                               "%d" % num_filters)

        self.num_filters = num_filters
        self.filter_size = filter_size[0]
        self.stride = stride[0]
        self.untie_biases = untie_biases
        self.dimshuffle = dimshuffle
        self.flip_filters = flip_filters
        self.partial_sum = partial_sum

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and "
                               "'pad'. To avoid ambiguity, please specify "
                               "only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = 0
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = 0
            elif border_mode == 'full':
                self.pad = self.filter_size - 1
            elif border_mode == 'same':
                # only works for odd filter size, but the even filter size case
                # is probably not worth supporting.
                self.pad = (self.filter_size - 1) // 2
            else:
                raise RuntimeError("Unsupported border_mode for "
                                   "Conv2DCCLayer: %s" % border_mode)
        else:
            self.pad = pad

        if W is None:
            if dimshuffle:
                W = init.GlorotUniform()
            else:
                W = init.GlorotUniform(c01b=True)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        elif self.untie_biases:
            if self.dimshuffle:
                biases_shape = (num_filters, self.output_shape[2],
                                self.output_shape[3])
            else:
                biases_shape = (num_filters, self.output_shape[1],
                                self.output_shape[2])
        else:
            biases_shape = (num_filters,)
        self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

        self.filter_acts_op = FilterActs(
            stride=self.stride, partial_sum=self.partial_sum, pad=self.pad)

    def get_W_shape(self):
        if self.dimshuffle:
            num_input_channels = self.input_shape[1]
            return (self.num_filters, num_input_channels, self.filter_size,
                    self.filter_size)
        else:
            num_input_channels = self.input_shape[0]
            return (num_input_channels, self.filter_size, self.filter_size,
                    self.num_filters)

    def get_output_shape_for(self, input_shape):
        if self.dimshuffle:
            batch_size = input_shape[0]
            input_rows, input_columns = input_shape[2:4]
        else:
            batch_size = input_shape[3]
            input_rows, input_columns = input_shape[1:3]

        output_rows = conv_output_length(input_rows,
                                         self.filter_size,
                                         self.stride,
                                         'pad', self.pad)

        output_columns = conv_output_length(input_columns,
                                            self.filter_size,
                                            self.stride,
                                            'pad', self.pad)

        if self.dimshuffle:
            return (batch_size, self.num_filters, output_rows, output_columns)
        else:
            return (self.num_filters, output_rows, output_columns, batch_size)

    def get_output_for(self, input, **kwargs):
        if self.dimshuffle:
            filters = self.W.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            input = input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
        else:
            filters = self.W

        if self.flip_filters:
            filters = filters[:, ::-1, ::-1, :]  # flip top-down, left-right

        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.stride != 1:
            # cuda-convnet calculates a non-standard strided output shape,
            # so we need to truncate the output in this case
            true_rows = conv_output_length(input.shape[1],
                                           self.filter_size,
                                           self.stride,
                                           'pad', self.pad)
            true_columns = conv_output_length(input.shape[2],
                                              self.filter_size,
                                              self.stride,
                                              'pad', self.pad)
            conved = conved[:, :true_rows, :true_columns, :]

        if self.b is not None:
            if self.untie_biases:
                biases = self.b.dimshuffle(0, 1, 2, 'x')  # c01 to c01b
            else:
                biases = self.b.dimshuffle(0, 'x', 'x', 'x')  # c to c01b
            conved += biases

        conved = self.nonlinearity(conved)

        if self.dimshuffle:
            return conved.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        else:
            return conved


class MaxPool2DCCLayer(CCLayer):
    def __init__(self, incoming, pool_size, ignore_border=False, stride=None,
                 dimshuffle=True, **kwargs):
        from pylearn2.sandbox.cuda_convnet.pool import MaxPool

        if 'pad' in kwargs:
            pad = kwargs.pop('pad')
            if as_tuple(pad, 2) != (0, 0):
                raise NotImplementedError("MaxPool2DCCLayer does not "
                                          "support padding")

        super(MaxPool2DCCLayer, self).__init__(incoming, **kwargs)

        pool_size = as_tuple(pool_size, 2)

        if pool_size[0] != pool_size[1]:
            raise NotImplementedError("MaxPool2DCCLayer only supports square "
                                      "pooling regions, but pool_size=(%d, %d)"
                                      % pool_size)

        self.pool_size = pool_size[0]

        if stride is None:
            self.stride = self.pool_size
        else:
            stride = as_tuple(stride, 2)
            if stride[0] != stride[1]:
                raise NotImplementedError("MaxPool2DCCLayer only supports "
                                          "using the same stride in both, "
                                          "directions but stride=(%d, %d)"
                                          % stride)
            self.stride = stride[0]

        if self.stride > self.pool_size:
            raise NotImplementedError("MaxPool2DCCLayer only supports "
                                      "stride <= pool_size.")

        # ignore_border argument is for compatibility with MaxPool2DLayer.
        # it is not supported. Borders are never ignored.
        if ignore_border is not False:
            raise NotImplementedError("MaxPool2DCCLayer does not support "
                                      "ignore_border.")

        self.dimshuffle = dimshuffle

        self.pool_op = MaxPool(ds=self.pool_size, stride=self.stride)

    def get_output_shape_for(self, input_shape):
        if self.dimshuffle:
            batch_size = input_shape[0]
            num_input_channels = input_shape[1]
            input_rows, input_columns = input_shape[2:4]
        else:
            batch_size = input_shape[3]
            num_input_channels = input_shape[0]
            input_rows, input_columns = input_shape[1:3]

        output_rows = int(np.ceil(float(input_rows - self.pool_size +
                                        self.stride) / self.stride))
        output_columns = int(np.ceil(float(input_columns - self.pool_size +
                                           self.stride) / self.stride))

        if self.dimshuffle:
            return (batch_size, num_input_channels, output_rows,
                    output_columns)
        else:
            return (num_input_channels, output_rows, output_columns,
                    batch_size)

    def get_output_for(self, input, **kwargs):
        if self.dimshuffle:
            input = input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b

        contiguous_input = gpu_contiguous(input)
        pooled = self.pool_op(contiguous_input)

        if self.dimshuffle:
            return pooled.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        else:
            return pooled


# TODO: crossmapnorm
# from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm


# Helper classes for switching between bc01 and c01b input formats

class ShuffleBC01ToC01BLayer(Layer):
    """
    This layer dimshuffles 4D input for interoperability between c01b and bc01
    ops.
    bc01 (theano) -> c01b (cuda-convnet)
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(1, 2, 3, 0)

bc01_to_c01b = ShuffleBC01ToC01BLayer  # shortcut


class ShuffleC01BToBC01Layer(Layer):
    """
    This layer dimshuffles 4D input for interoperability between c01b and bc01
    ops.
    c01b (cuda-convnet) -> bc01 (theano)
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[3], input_shape[0], input_shape[1], input_shape[2])

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(3, 0, 1, 2)

c01b_to_bc01 = ShuffleC01BToBC01Layer  # shortcut


# c01b versions of other Layer classes

class NINLayer_c01b(Layer):
    """
    This does the same as lasagne.layers.NINLayer, but operates with c01b
    axis arrangement instead of bc01. This reduces the number of shuffles
    and reshapes required and might be faster as a result.
    """
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(c01b=True), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer_c01b, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[0]

        self.W = self.add_param(W, (num_units, num_input_channels), name="W")
        if b is None:
            self.b = None
        elif self.untie_biases:
            biases_shape = (num_units,) + self.output_shape[1:-1]
        else:
            biases_shape = (num_units,)
        self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (self.num_units,) + input_shape[1:]

    def get_output_for(self, input, **kwargs):
        # fc * c01b... = f01b...
        out = T.tensordot(self.W, input, axes=[[1], [0]])

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                bias_axes = range(input.ndim - 1) + ['x']
            else:
                bias_axes = [0] + (['x'] * (input.ndim - 1))
            b_shuffled = self.b.dimshuffle(bias_axes)
            activation = out + b_shuffled

        return self.nonlinearity(activation)
