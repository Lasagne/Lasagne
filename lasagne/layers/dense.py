import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer


__all__ = [
    "DenseLayer",
    "NINLayer",
]


class DenseLayer(Layer):
    """
    A fully connected layer.

    :parameters:
        - input_layer : `Layer` instance
            The layer from which this layer will obtain its input

        - num_units : int
            The number of units of the layer

        - W : Theano shared variable, numpy array or callable
            An initializer for the weights of the layer. If a Theano shared
            variable is provided, it is used unchanged. If a numpy array is
            provided, a shared variable is created and initialized with the
            array. If a callable is provided, a shared variable is created and
            the callable is called with the desired shape to generate suitable
            initial values. The variable is then initialized with those values.

        - b : Theano shared variable, numpy array, callable or None
            An initializer for the biases of the layer. If a Theano shared
            variable is provided, it is used unchanged. If a numpy array is
            provided, a shared variable is created and initialized with the
            array. If a callable is provided, a shared variable is created and
            the callable is called with the desired shape to generate suitable
            initial values. The variable is then initialized with those values.

            If None is provided, the layer will have no biases.

        - nonlinearity : callable or None
            The nonlinearity that is applied to the layer activations. If None
            is provided, the layer will be linear.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.create_param(W, (num_inputs, num_units), name="W")
        self.b = (self.create_param(b, (num_units,), name="b")
                  if b is not None else None)

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
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
    Like DenseLayer, but broadcasting across all trailing dimensions beyond the
    2nd.  This results in a convolution operation with filter size 1 on all
    trailing dimensions.  Any number of trailing dimensions is supported,
    so NINLayer can be used to implement 1D, 2D, 3D, ... convolutions.
    """
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[1]

        self.W = self.create_param(W, (num_input_channels, num_units),
                                   name="W")
        if b is None:
            self.b = None
        elif self.untie_biases:
            output_shape = self.get_output_shape()
            self.b = self.create_param(b, (num_units,) + output_shape[2:],
                                       name="b")
        else:
            self.b = self.create_param(b, (num_units,), name="b")

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units) + input_shape[2:]

    def get_output_for(self, input, **kwargs):
        # cf * bc01... = fb01...
        out_r = T.tensordot(self.W, input, axes=[[0], [1]])
        # input dims to broadcast over
        remaining_dims = range(2, input.ndim)
        # bf01...
        out = out_r.dimshuffle(1, 0, *remaining_dims)

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                # no broadcast
                remaining_dims_biases = range(1, input.ndim - 1)
            else:
                remaining_dims_biases = ['x'] * (input.ndim - 2)  # broadcast
            b_shuffled = self.b.dimshuffle('x', 0, *remaining_dims_biases)
            activation = out + b_shuffled

        return self.nonlinearity(activation)
