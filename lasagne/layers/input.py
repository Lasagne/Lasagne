import numpy as np
import theano
import theano.tensor as T

from .. import utils

from .base import Layer


__all__ = [
    "InputLayer",
]


class InputLayer(Layer):
    def __init__(self, shape, input_var=None):
        self.shape = shape
        ndim = len(shape)
        if input_var is None:
            # create the right TensorType for the given number of dimensions
            input_var_type = T.TensorType(theano.config.floatX, [False] * ndim)
            input_var = input_var_type("input")
        else:
            # ensure the given variable has the correct dimensionality
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, "
                    "but variable has %d" % (ndim, input_var.ndim))
        self.input_var = input_var

    def get_output_shape(self):
        return self.shape

    def get_output(self, input=None, *args, **kwargs):
        if isinstance(input, dict):
            input = input.get(self, None)
        if input is None:
            input = self.input_var
        return utils.as_theano_expression(input)