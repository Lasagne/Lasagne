import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer

from theano.sandbox.cuda.dnn import dnn_pool


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
        return dnn_pool(input, self.ds, self.strides, self.mode)


class MaxPool2DDNNLayer(Pool2DDNNLayer): # for consistency
    def __init__(self, input_layer, ds, strides=None):
        super(MaxPool2DDNNLayer, self).__init__(input_layer, ds, strides, mode='max')
