import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer


dnn_available = False

if theano.config.device.startswith("gpu"):
    from theano.sandbox.cuda import dnn
    if dnn.dnn_available():
        dnn_available = True
 

__all__ = [
    "Pool2DDNNLayer",
    "MaxPool2DDNNLayer",
]


class DNNLayer(Layer):
    pass


class Pool2DDNNLayer(DNNLayer):
    def __init__(self, input_layer, ds, strides=None, mode='max'):
        super(Pool2DDNNLayer, self).__init__(input_layer)
        self.ds = ds # a tuple
        self.mode = mode
        self.strides = strides if strides is not None else ds
        
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape) # copy / convert to mutable list
        output_shape[2] = (output_shape[2] - self.ds[0]) // self.strides[0] + 1
        output_shape[3] = (output_shape[3] - self.ds[1]) // self.strides[1] + 1
        return tuple(output_shape)

    def get_output_for(self, input, *args, **kwargs):
        if not dnn_available:
            raise RuntimeError("cudnn is not available.")
        return dnn.dnn_pool(input, self.ds, self.strides, self.mode)


class MaxPool2DDNNLayer(Pool2DDNNLayer): # for consistency
    def __init__(self, input_layer, ds, strides=None):
        super(MaxPool2DDNNLayer, self).__init__(input_layer, ds, strides, mode='max')
