import numpy as np
import theano
import theano.tensor as T

from ..theano_extensions import padding

from .base import Layer


__all__ = [
    "FlattenLayer",
    "flatten",
    "PadLayer",
    "pad",
]



class FlattenLayer(Layer):
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], int(np.prod(input_shape[1:])))

    def get_output_for(self, input, *args, **kwargs):
        return input.flatten(2)

flatten = FlattenLayer # shortcut


class PadLayer(Layer):
    def __init__(self, input_layer, width, val=0, batch_ndim=2, **kwargs):
        super(PadLayer, self).__init__(input_layer, **kwargs)
        self.width = width
        self.val = val
        self.batch_ndim = batch_ndim

    def get_output_shape_for(self, input_shape):
        output_shape = ()
        for k, s in enumerate(input_shape):
            if k < self.batch_ndim:
                output_shape += (s,)
            else:
                output_shape += (s + 2 * self.width,)

        return output_shape

    def get_output_for(self, input, *args, **kwargs):
        return padding.pad(input, self.width, self.val, self.batch_ndim)

pad = PadLayer # shortcut