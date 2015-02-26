import theano
import theano.tensor as T

from .. import utils

from .base import Layer


__all__ = [
    "InputLayer",
]


class InputLayer(Layer):
    """
    This layer holds a symbolic variable that represents a network input. A
    variable can be specified when the layer is instantiated, else it is
    created.

    :parameters:
        - shape : tuple of int
            The shape of the input

        - input_var : Theano symbolic variable or None (default: None)
            A variable representing a network input. If it is not provided,
            a variable will be created.

    :usage:
        >>> from lasagne.layers import InputLayer
        >>> l_in = InputLayer((100, 20))
    """
    def __init__(self, shape, input_var=None, name=None, **kwargs):
        self.shape = shape
        ndim = len(shape)
        if input_var is None:
            # create the right TensorType for the given number of dimensions
            input_var_type = T.TensorType(theano.config.floatX, [False] * ndim)
            var_name = ("%s.input" % name) if name is not None else "input"
            input_var = input_var_type(var_name)
        else:
            # ensure the given variable has the correct dimensionality
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, but variable has "
                                 "%d" % (ndim, input_var.ndim))
        self.input_var = input_var
        self.name = name

    def get_output_shape(self):
        return self.shape

    def get_output(self, input=None, *args, **kwargs):
        if isinstance(input, dict):
            input = input.get(self, None)
        if input is None:
            input = self.input_var
        return utils.as_theano_expression(input)
