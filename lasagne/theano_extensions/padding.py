"""
Padding
"""

import theano.tensor as T


def pad(x, width, val=0, batch_ndim=1):
    """
    Pad a tensor with a constant value.

    Parameters
    ----------
    x : tensor

    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.

    val : float
        The constant value used for padding

    batch_ndim : integer
        Dimensions before the value will not be padded.

    """
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):
        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] += l + r
        indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

    if val:
        out = T.ones(output_shape) * val
    else:
        out = T.zeros(output_shape)
    return T.set_subtensor(out[tuple(indices)], x)
