import numpy as np
import theano
import theano.tensor as T

from ..theano_extensions import padding

from .base import Layer


__all__ = [
    "FlattenLayer",
    "flatten",
    "ReshapeLayer",
    "PadLayer",
    "pad",
]



class FlattenLayer(Layer):
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], int(np.prod(input_shape[1:])))

    def get_output_for(self, input, *args, **kwargs):
        return input.flatten(2)

flatten = FlattenLayer # shortcut


class ReshapeLayer(Layer):
    """
    A layer reshaping its input tensor to another tensor of the same total
    number of elements.

    :parameters:
        - incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape

        - shape : tuple
            The target shape. Any of its elements can be `None`, denoting to
            retain the size of the input shape for this dimension. At most one
            element can be `-1`, denoting to infer the size for this dimension
            to match the total number of elements of the input shape. Any
            remaining elements must be positive integers.

    :usage:
        >>> from lasagne.layers import InputLayer, ReshapeLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = ReshapeLayer(l_in, (None, 2, 10))
        >>> l1.get_output_shape()
        (100, 2, 10)
        >>> l2 = ReshapeLayer(l_in, (None, 1, 2, 5, -1))
        >>> l2.get_output_shape()
        (100, 1, 2, 5, 2)

    :note:
        The tensor elements will be fetched and placed in C-like order. That
        is, reshaping `[1,2,3,4,5,6]` to shape `(2,3)` will result in a matrix
        `[[1,2,3],[4,5,6]]`, not in `[[1,3,5],[2,4,6]]` (Fortran-like order),
        regardless of the memory layout of the input tensor. For C-contiguous
        input, reshaping is cheap, for others it may require copying the data.
    """

    def __init__(self, incoming, shape):
        super(ReshapeLayer, self).__init__(incoming)
        shape = tuple(shape)
        if not all(s is None or isinstance(s, int) for s in shape):
            raise ValueError("`shape` must be a tuple of int and/or None")
        if any(s is not None and (s == 0 or s < -1) for s in shape):
            raise ValueError("`shape` integers must be positive or -1")
        if sum(s == -1 for s in shape) > 1:
            raise ValueError("`shape` cannot contain multiple -1")
        self.shape = shape

    def get_output_shape_for(self, input_shape, *args, **kwargs):
        # First, replace all `None` with the corresponding input dimension
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if o is None:
                output_shape[dim] = input_shape[dim]
        # Secondly, infer value for -1 if needed
        if -1 in output_shape:
            dim = output_shape.index(-1)
            output_shape[dim] = np.prod(input_shape) // -np.prod(output_shape)
        # Sanity check
        if np.prod(input_shape) != np.prod(output_shape):
            raise ValueError("%s cannot be reshaped to specification %s. "
                             "The total size mismatches." %
                             (input_shape, self.shape))
        return tuple(output_shape)

    def get_output_for(self, input, *args, **kwargs):
        # Replace all `None` with the corresponding input dimension
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if o is None:
                output_shape[dim] = input.shape[dim]
        # Everything else is handled by Theano
        return input.reshape(tuple(output_shape))


class PadLayer(Layer):
    def __init__(self, incoming, width, val=0, batch_ndim=2, **kwargs):
        super(PadLayer, self).__init__(incoming, **kwargs)
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