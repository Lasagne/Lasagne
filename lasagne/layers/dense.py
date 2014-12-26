import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer


__all__ = [
    "DenseLayer",
    "NINLayer",
]


class DenseLayer(Layer):
    def __init__(self, input_layer, num_units, W=init.Uniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify):
        super(DenseLayer, self).__init__(input_layer)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units

        output_shape = self.input_layer.get_output_shape()
        num_inputs = int(np.prod(output_shape[1:]))

        self.W = self.create_param(W, (num_inputs, num_units))
        self.b = self.create_param(b, (num_units,)) if b is not None else None

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class NINLayer(Layer):
    """
    Network-in-network layer.
    Like DenseLayer, but broadcasting across all trailing dimensions beyond the 2nd.
    This results in a convolution operation with filter size 1 on all trailing dimensions.
    Any number of trailing dimensions is supported, so NINLayer can be used to implement
    1D, 2D, 3D, ... convolutions.
    """
    def __init__(self, input_layer, num_units, untie_biases=False,
        W=init.Uniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify):
        super(NINLayer, self).__init__(input_layer)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.untie_biases = untie_biases

        output_shape = self.input_layer.get_output_shape()
        num_input_channels = output_shape[1]

        self.W = self.create_param(W, (num_input_channels, num_units))
        if b is None:
            self.b = None
        elif self.untie_biases:
            output_shape = self.get_output_shape()
            self.b = self.create_param(b, (num_units,) + output_shape[2:])
        else:
            self.b = self.create_param(b, (num_units,))

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units) + input_shape[2:]

    def get_output_for(self, input, *args, **kwargs):
        out_r = T.tensordot(self.W, input, axes=[[0], [1]]) # cf * bc01... = fb01...
        remaining_dims = range(2, input.ndim) # input dims to broadcast over
        out = out_r.dimshuffle(1, 0, *remaining_dims) # bf01...

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                remaining_dims_biases = range(1, input.ndim - 1) # no broadcast
            else:
                remaining_dims_biases = ['x'] * (input.ndim - 2) # broadcast
            b_shuffled = self.b.dimshuffle('x', 0, *remaining_dims_biases)
            activation = out + b_shuffled

        return self.nonlinearity(activation)
