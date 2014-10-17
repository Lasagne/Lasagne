import numpy as np

import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities
from . import base

from theano.sandbox.cuda.basic_ops import gpu_contiguous

# TODO: make sure to document the limitations and 'best practices' (i.e. minibatch size % 128 == 0)
# TODO: see if the 'dimshuffle' logic can be put in the base class instead.


# base class for all layers that use ops from pylearn2.sandbox.cuda_convnet
class CCLayer(base.Layer):
    pass


class Conv2DCCLayer(CCLayer):
    def __init__(self, input_layer, num_filters, filter_size, strides=(1, 1), border_mode=None, untie_biases=False,
                 W=init.Uniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify, pad=None,
                 dimshuffle=True, flip_filters=False, partial_sum=1):
        from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs

        super(Conv2DCCLayer, self).__init__(input_layer)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if filter_size[0] != filter_size[1]:
            raise RuntimeError("Conv2DCCLayer only supports square filters, but filter_size=(%d, %d)" % filter_size)

        if strides[0] != strides[1]:
            raise RuntimeError("Conv2DCCLayer only supports square strides, but strides=(%d, %d)" % strides)

        if num_filters % 16 != 0:
            raise RuntimeError("Conv2DCCLayer requires num_filters to be a multiple of 16, but num_filters is %d" % num_filters)

        self.num_filters = num_filters
        self.filter_size = filter_size[0]
        self.stride = strides[0]
        self.untie_biases = untie_biases
        self.dimshuffle = dimshuffle
        self.flip_filters = flip_filters
        self.partial_sum = partial_sum

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and 'pad'. To avoid ambiguity, please specify only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = 0
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = 0
            elif border_mode == 'full':
                self.pad = self.filter_size - 1
            elif border_mode == 'same':
                # only works for odd filter size, but the even filter size case is probably not worth supporting.
                self.pad = (self.filter_size - 1) // 2
            else:
                raise RuntimeError("Unsupported border_mode for Conv2DCCLayer: %s" % border_mode)
        else:
            self.pad = pad

        self.W = self.create_param(W, self.get_W_shape())
        if self.untie_biases:
            output_shape = self.get_output_shape()
            if self.dimshuffle:
                self.b = self.create_param(b, (num_filters, output_shape[2], output_shape[3]))
            else:
                self.b = self.create_param(b, (num_filters, output_shape[1], output_shape[2]))
        else:
            self.b = self.create_param(b, (num_filters,))

        self.filter_acts_op = FilterActs(stride=self.stride, partial_sum=self.partial_sum, pad=self.pad)

    def get_W_shape(self):
        if self.dimshuffle:
            num_input_channels = self.input_layer.get_output_shape()[1]
            return (self.num_filters, num_input_channels, self.filter_size, self.filter_size)
        else:
            num_input_channels = self.input_layer.get_output_shape()[0]
            return (num_input_channels, self.filter_size, self.filter_size, self.num_filters)

    def get_params(self):
        return [self.W, self.b]

    def get_bias_params(self):
        return [self.b]

    def get_output_shape_for(self, input_shape):
        if self.dimshuffle:
            batch_size = input_shape[0]
            input_width, input_height = input_shape[2:4]
        else:
            batch_size = input_shape[3]
            input_width, input_height = input_shape[1:3]

        output_width = (input_width + 2*self.pad - self.filter_size) // self.stride + 1
        output_height = (input_height + 2*self.pad - self.filter_size) // self.stride + 1

        if self.dimshuffle:
            return (batch_size, self.num_filters, output_width, output_height)
        else:
            return (self.num_filters, output_width, output_height, batch_size)

    def get_output_for(self, input, *args, **kwargs):
        if self.dimshuffle:
            filters = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
            input = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        else:
            filters = self.W

        if self.flip_filters:
            filters = filters[:, ::-1, ::-1, :] # flip width, height
        
        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.untie_biases:
            biases = self.b.dimshuffle(0, 1, 2, 'x') # c01 to c01b
        else:
            biases = self.b.dimshuffle(0, 'x', 'x', 'x') # c to c01b

        conved += biases
        conved = self.nonlinearity(conved)

        if self.dimshuffle:
            return conved.dimshuffle(3, 0, 1, 2) # c01b to bc01
        else:
            return conved


class MaxPool2DCCLayer(CCLayer):
    def __init__(self, input_layer, ds, ignore_border=False, strides=None, dimshuffle=True):
        from pylearn2.sandbox.cuda_convnet.pool import MaxPool

        super(MaxPool2DCCLayer, self).__init__(input_layer)
        if ds[0] != ds[1]:
            raise RuntimeError("MaxPool2DCCLayer only supports square pooling regions, but ds=(%d, %d)" % ds)

        if strides is not None and strides[0] != strides[1]:
            raise RuntimeError("MaxPool2DCCLayer only supports using the same stride in both directions, but strides=(%d, %d)" % strides)

        # ignore_border argument is for compatibility with MaxPool2DLayer.
        # it is not supported. Borders are never ignored.
        if ignore_border != False:
            raise RuntimeError("MaxPool2DCCLayer does not support ignore_border.")

        self.ds = ds[0]
        if strides is None:
            self.stride = self.ds
        else:
            self.stride = strides[0]
        self.dimshuffle = dimshuffle

        self.pool_op = MaxPool(ds=self.ds, stride=self.stride)

    def get_output_shape_for(self, input_shape):
        if self.dimshuffle:
            batch_size = input_shape[0]
            num_input_channels = input_shape[1]
            input_width, input_height = input_shape[2:4]
        else:
            batch_size = input_shape[3]
            num_input_channels = input_shape[0]
            input_width, input_height = input_shape[1:3]

        output_width = int(np.ceil(float(input_width - self.ds + self.stride) / self.stride))
        output_height = int(np.ceil(float(input_height - self.ds + self.stride) / self.stride))
        
        if self.dimshuffle:
            return (batch_size, num_input_channels, output_width, output_height)
        else:
            return (num_input_channels, output_width, output_height, batch_size)

    def get_output_for(self, input, *args, **kwargs):
        if self.dimshuffle:
            input = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b

        contiguous_input = gpu_contiguous(input)
        pooled = self.pool_op(contiguous_input)

        if self.dimshuffle:
            return pooled.dimshuffle(3, 0, 1, 2) # c01b to bc01
        else:
            return pooled


# TODO: crossmapnorm
# from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm


## Helper classes for switching between bc01 and c01b input formats

class ShuffleBC01ToC01BLayer(base.Layer):
    """
    This layer dimshuffles 4D input for interoperability between c01b and bc01 ops.
    bc01 (theano) -> c01b (cuda-convnet)
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

    def get_output_for(self, input, *args, **kwargs):
        return input.dimshuffle(1, 2, 3, 0)

bc01_to_c01b = ShuffleBC01ToC01BLayer # shortcut


class ShuffleC01BToBC01Layer(base.Layer):
    """
    This layer dimshuffles 4D input for interoperability between c01b and bc01 ops.
    c01b (cuda-convnet) -> bc01 (theano)
    """
    def get_output_shape_for(self, input_shape):
        return (input_shape[3], input_shape[0], input_shape[1], input_shape[2])

    def get_output_for(self, input, *args, **kwargs):
        return input.dimshuffle(3, 0, 1, 2)

c01b_to_bc01 = ShuffleC01BToBC01Layer # shortcut


## c01b versions of other Layer classes

class NINLayer_c01b(base.Layer):
    """
    This does the same as nntools.layers.NINLayer, but operates with c01b
    axis arrangement instead of bc01. This reduces the number of shuffles
    and reshapes required and might be faster as a result.
    """
    def __init__(self, input_layer, num_units, untie_biases=False,
        W=init.Uniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify):
        super(NINLayer_c01b, self).__init__(input_layer)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.untie_biases = untie_biases

        output_shape = self.input_layer.get_output_shape()
        num_input_channels = output_shape[0]

        self.W = self.create_param(W, (num_units, num_input_channels))
        if self.untie_biases:
            output_shape = self.get_output_shape()
            self.b = self.create_param(b, (num_units,) + output_shape[1:-1])
        else:
            self.b = self.create_param(b, (num_units,))

    def get_params(self):
        return [self.W, self.b]

    def get_bias_params(self):
        return [self.b]

    def get_output_shape_for(self, input_shape):
        return (self.num_units,) + input_shape[1:]

    def get_output_for(self, input, *args, **kwargs):
        out = T.tensordot(self.W, input, axes=[[1], [0]]) # fc * c01b... = f01b...

        if self.untie_biases:
            bias_axes = range(input.ndim - 1) + ['x']
        else:
            bias_axes = [0] + (['x'] * (input.ndim - 1))
        b_shuffled = self.b.dimshuffle(bias_axes)

        return self.nonlinearity(out + b_shuffled)
