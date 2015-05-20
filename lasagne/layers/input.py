from collections import OrderedDict

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

    Parameters
    ----------
    shape : tuple of `int` or `None` elements
        The shape of the input. Any element can be `None` to indicate that the
        size of that dimension is not fixed at compile time.

    input_var : Theano symbolic variable or `None` (default: `None`)
        A variable representing a network input. If it is not provided, a
        variable will be created.

    Raises
    ------
    ValueError
        If the dimension of `input_var` is not equal to `len(shape)`

    Notes
    -----
    The first dimension usually indicates the batch size. If you specify it,
    Theano may apply more optimizations while compiling the training or
    prediction function, but the compiled function will not accept data of a
    different batch size at runtime. To compile for a variable batch size, set
    the first shape element to `None` instead.

    Examples
    --------
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
        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape
