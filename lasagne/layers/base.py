import numpy as np

import theano
import theano.tensor as T

from .. import utils


__all__ = [
    "Layer",
    "MultipleInputsLayer",
]


## Layer base class

class Layer(object):
    def __init__(self, input_layer):
        self.input_layer = input_layer

    def get_params(self):
        """
        Get all Theano variables that parameterize the layer.
        """
        return []

    def get_bias_params(self):
        """
        Get all Theano variables that are bias parameters for the layer.
        """
        return []

    def get_output_shape(self):
        input_shape = self.input_layer.get_output_shape()
        return self.get_output_shape_for(input_shape)

    def get_output(self, input=None, *args, **kwargs):
        """
        Computes the output of the network at this layer. Optionally, you can
        define an input to propagate through the network instead of using the
        input variables associated with the network's input layers.

        :parameters:
            - input : None, Theano expression, numpy array, or dict
                If None, uses the inputs of the :class:`InputLayer` instances.
                If a Theano expression, this will replace the inputs of all
                :class:`InputLayer` instances (useful if your network has a
                single input layer).
                If a numpy array, this will be wrapped as a Theano constant
                and used just like a Theano expression.
                If a dictionary, any :class:`Layer` instance (including the
                input layers) can be mapped to a Theano expression or numpy
                array to use instead of its regular output.

        :returns:
            - output : Theano expression
                the output of this layer given the input to the network

        :note:
            When implementing a new :class:`Layer` class, you will usually
            keep this unchanged and just override `get_output_for()`.
        """
        if isinstance(input, dict) and (self in input):
            # this layer is mapped to an expression or numpy array
            return utils.as_theano_expression(input[self])
        else: # in all other cases, just pass the network input on to the next layer.
            layer_input = self.input_layer.get_output(input, *args, **kwargs)
            return self.get_output_for(layer_input, *args, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape # By default, the shape is assumed to be preserved.
        # This means that layers performing elementwise operations, or other
        # shape-preserving operations (such as normalization), only need to
        # implement a single method, i.e. get_output_for().

    def get_output_for(self, input, *args, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).

        :parameters:
            - input : Theano expression
                the expression to propagate through this layer

        :returns:
            - output : Theano expression
                the output of this layer given the input to this layer

        :note:
            This is called by the base :class:`Layer` implementation to
            propagate data through a network in `get_output()`. While
            `get_output()` asks the underlying layers for input and thus
            returns an expression for a layer's output in terms of the
            network's input, `get_output_for()` just performs a single step
            and returns an expression for a layer's output in terms of
            that layer's input.
        """
        raise NotImplementedError

    @staticmethod
    def create_param(param, shape):
        """
        Helper method to create Theano shared variables for
        Layer parameters and to initialize them.

        param: one of three things:
            - a numpy array with the initial parameter values
            - a Theano shared variable
            - a function or callable that takes the desired
              shape of the parameter array as its single
              argument.

        shape: the desired shape of the parameter array.
        """
        if isinstance(param, np.ndarray):
            if param.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be %s" % (param.shape, shape))
            return theano.shared(param)

        elif isinstance(param, theano.compile.SharedVariable):
            # cannot check shape here, the shared variable might not be initialized correctly yet.
            return param

        elif hasattr(param, '__call__'):
            arr = param(shape)
            if not isinstance(arr, np.ndarray):
                raise RuntimeError("cannot initialize parameters: the provided callable did not return a numpy array")

            return theano.shared(utils.floatX(arr))

        else:
            raise RuntimeError("cannot initialize parameters: 'param' is not a numpy array, a Theano shared variable, or a callable")


class MultipleInputsLayer(Layer):
    def __init__(self, input_layers):
        self.input_layers = input_layers

    def get_output_shape(self):
        input_shapes = [input_layer.get_output_shape() for input_layer in self.input_layers]
        return self.get_output_shape_for(input_shapes)

    def get_output(self, input=None, *args, **kwargs):
        if isinstance(input, dict) and (self in input):
            # this layer is mapped to an expression or numpy array
            return utils.as_theano_expression(input[self])
        else: # in all other cases, just pass the network input on to the next layers.
            layer_inputs = [input_layer.get_output(input, *args, **kwargs) for input_layer in self.input_layers]
            return self.get_output_for(layer_inputs, *args, **kwargs)

    def get_output_shape_for(self, input_shapes):
        raise NotImplementedError

    def get_output_for(self, inputs, *args, **kwargs):
        raise NotImplementedError

