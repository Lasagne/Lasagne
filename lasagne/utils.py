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
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))


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


def compute_norms(array, norm_axes=None):
    """
    Compute incoming weight vector norms.

    :parameters:
        - array : ndarray
            Weight array
        - norm_axes : sequence (list or tuple)
            The axes over which to compute the norm.  This overrides the
            default norm axes defined for the number of dimensions
            in `array`. When this is not specified and `array` is a 2D array,
            this is set to `(0,)`. If `array` is a 3D, 4D or 5D array, it is
            set to a tuple listing all axes but axis 0. The former default is
            useful for working with dense layers, the latter is useful for 1D,
            2D and 3D convolutional layers.
            (Optional)

    :returns:
        - norms : 1D array
            1D array of incoming weight vector norms.
    :usage:
        >>> array = np.random.randn(100, 200)
        >>> norms = compute_norms(array)
        >>> norms.shape
        (200,)

        >>> norms = compute_norms(array, norm_axes=(1,))
        >>> norms.shape
        (100,)

    """
    ndim = array.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(array.ndim)
        )

    norms = np.sqrt(np.sum(array**2, axis=sum_over))

    return norms
