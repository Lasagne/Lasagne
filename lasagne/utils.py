import numpy as np

import theano
import theano.tensor as T


def floatX(arr):
    """
    Shortcut to turn a numpy array into an array with the
    correct dtype for Theano.
    """
    return arr.astype(theano.config.floatX)


def shared_empty(dim=2, dtype=None):
    """
    Shortcut to create an empty Theano shared variable with
    the specified number of dimensions.
    """
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))


def as_theano_expression(input):
    """
    Wraps the given input as a Theano constant if it is not
    a valid Theano expression already. Useful to transparently
    handle numpy arrays and Python scalars, for example.
    """
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano "
                    "expression and cannot be wrapped as a Theano "
                    "constant (original exception: %s)" %
                    (type(input), e))


def one_hot(x, m=None):
    """
    Given a vector of integers from 0 to m-1, returns a matrix
    with a one-hot representation, where each row corresponds
    to an element of x.
    """
    if m is None:
        m = T.cast(T.max(x) + 1, 'int32')

    return T.eye(m)[T.cast(x, 'int32')]


def unique(l):
    """
    Create a new list from l with duplicate entries removed,
    while preserving the original order.
    """
    new_list = []
    for el in l:
        if el not in new_list:
            new_list.append(el)

    return new_list


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.

    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.

    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)

    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.

    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    if axis < 0:
        axis += tensor_list[0].ndim

    concat_size = sum(tensor.shape[axis] for tensor in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tensor in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tensor.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tensor)
        offset += tensor.shape[axis]

    return out
