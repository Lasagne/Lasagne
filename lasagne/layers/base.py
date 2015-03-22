import numpy as np

import theano

from .. import utils


__all__ = [
    "Layer",
    "MultipleInputsLayer",
]


# Layer base class

class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network.
    It should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the
    full network.
    """
    def __init__(self, incoming, name=None):
        """
        Instantiates the layer.

        :parameters:
            - incoming : a :class:`Layer` instance or a tuple
                the layer feeding into this layer, or the expected input shape
            - name : a string or None
                an optional name to attach to this layer
        """
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.get_output_shape()
            self.input_layer = incoming
        self.name = name

    def get_params(self):
        """
        Returns a list of all the Theano variables that parameterize the
        layer.

        :returns:
            - list
                the list of Theano variables.

        :note:
            By default this returns an empty list, but it should be overridden
            in a subclass that has trainable parameters.
        """
        return []

    def get_bias_params(self):
        """
        Returns a list of all the Theano variables that are bias parameters
        for the layer.

        :returns:
            - bias_params : list
                the list of Theano variables.

        :note:
            By default this returns an empty list, but it should be overridden
            in a subclass that has trainable parameters.

            While `get_params()` should return all Theano variables,
            `get_bias_params()` should only return those corresponding to bias
            parameters. This is useful when specifying regularization (it is
            often undesirable to regularize bias parameters).
        """
        return []

    def get_output_shape(self):
        """
        Computes the output shape of the network at this layer.

        :returns:
            - output shape: tuple
                a tuple that represents the output shape of this layer. The
                tuple has as many elements as there are output dimensions, and
                the elements of the tuple are either integers or `None`.

        :note:
            When implementing a new :class:`Layer` class, you will usually
            keep this unchanged and just override `get_output_shape_for()`.
        """
        return self.get_output_shape_for(self.input_shape)

    def get_output(self, input=None, **kwargs):
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
        elif self.input_layer is None:
            raise RuntimeError("get_output() called on a free-floating layer; "
                               "there isn't anything to get its input from. "
                               "Did you mean get_output_for()?")
        else:  # in all other cases, just pass the input on to the next layer.
            layer_input = self.input_layer.get_output(input, **kwargs)
            return self.get_output_for(layer_input, **kwargs)

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of this layer, given an input shape.

        :parameters:
            - input_shape : tuple
                a tuple representing the shape of the input. The tuple should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.

        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.

        :note:
            This method will typically be overridden when implementing a new
            :class:`Layer` class. By default it simply returns the input
            shape. This means that a layer that does not modify the shape
            (e.g. because it applies an elementwise operation) does not need
            to override this method.
        """
        return input_shape

    def get_output_for(self, input, **kwargs):
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

            This method should be overridden when implementing a new
            :class:`Layer` class. By default it raises `NotImplementedError`.
        """
        raise NotImplementedError

    def create_param(self, param, shape, name=None):
        """
        Helper method to create Theano shared variables for layer parameters
        and to initialize them.

        :parameters:
            - param : numpy array, Theano shared variable, or callable
                One of three things:
                    * a numpy array with the initial parameter values
                    * a Theano shared variable representing the parameters
                    * a function or callable that takes the desired shape of
                      the parameter array as its single argument.

            - shape : tuple
                a tuple of integers representing the desired shape of the
                parameter array.

        :returns:
            - variable : Theano shared variable
                a Theano shared variable representing layer parameters. If a
                numpy array was provided, the variable is initialized to
                contain this array. If a shared variable was provided, it is
                simply returned. If a callable was provided, it is called, and
                its output is used to initialize the variable.

        :note:
            This method should be used in `__init__()` when creating a
            :class:`Layer` subclass that has trainable parameters. This
            enables the layer to support initialization with numpy arrays,
            existing Theano shared variables, and callables for generating
            initial parameter values.
        """
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)

        if isinstance(param, theano.compile.SharedVariable):
            # We cannot check the shape here, the shared variable might not be
            # initialized correctly yet. We can check the dimensionality
            # though. Note that we cannot assign a name here.
            if param.ndim != len(shape):
                raise RuntimeError("shared variable has %d dimensions, "
                                   "should be %d" % (param.ndim, len(shape)))
            return param

        elif isinstance(param, np.ndarray):
            if param.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be "
                                   "%s" % (param.shape, shape))
            return theano.shared(param, name=name)

        elif hasattr(param, '__call__'):
            arr = param(shape)
            if not isinstance(arr, np.ndarray):
                raise RuntimeError("cannot initialize parameters: the "
                                   "provided callable did not return a numpy "
                                   "array")

            return theano.shared(utils.floatX(arr), name=name)

        else:
            raise RuntimeError("cannot initialize parameters: 'param' is not "
                               "a numpy array, a Theano shared variable, or a "
                               "callable")


class MultipleInputsLayer(Layer):
    """
    This class represents a layer that aggregates input from multiple layers.
    It should be subclassed when implementing new types of layers that
    obtain their input from multiple layers.
    """
    def __init__(self, incomings, name=None):
        """
        Instantiates the layer.

        :parameters:
            - incomings : a list of :class:`Layer` instances or tuples
                the layers feeding into this layer, or expected input shapes
            - name : a string or None
                an optional name to attach to this layer
        """
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.get_output_shape()
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name

    def get_output_shape(self):
        return self.get_output_shape_for(self.input_shapes)

    def get_output(self, input=None, **kwargs):
        if isinstance(input, dict) and (self in input):
            # this layer is mapped to an expression or numpy array
            return utils.as_theano_expression(input[self])
        elif any(input_layer is None for input_layer in self.input_layers):
            raise RuntimeError("get_output() called on a free-floating layer; "
                               "there isn't anything to get its inputs from. "
                               "Did you mean get_output_for()?")
        # In all other cases, just pass the network input on to the next layers
        else:
            layer_inputs = [input_layer.get_output(input, **kwargs) for
                            input_layer in self.input_layers]
            return self.get_output_for(layer_inputs, **kwargs)

    def get_output_shape_for(self, input_shapes):
        """
        Computes the output shape of this layer, given a list of input shapes.

        :parameters:
            - input_shape : list of tuple
                a list of tuples, with each tuple representing the shape of
                one of the inputs (in the correct order). These tuples should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.

        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.

        :note:
            This method must be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).

        :parameters:
            - inputs : list of Theano expressions
                The Theano expressions to propagate through this layer

        :returns:
            - output : Theano expressions
                the output of this layer given the inputs to this layer

        :note:
            This is called by the base :class:`MultipleInputsLayer`
            implementation to propagate data through a network in
            `get_output()`. While `get_output()` asks the underlying layers
            for input and thus returns an expression for a layer's output in
            terms of the network's input, `get_output_for()` just performs a
            single step and returns an expression for a layer's output in
            terms of that layer's input.

            This method should be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError
