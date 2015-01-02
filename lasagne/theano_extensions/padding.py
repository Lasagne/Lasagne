"""
Padding
"""

import numpy as np

import theano
import theano.tensor as T


def pad(x, width, val=0, batch_ndim=1):
    """
    pad all dimensions except the first 'batch_ndim' with 'width'
    zeros on both sides, or with another value specified in 'val'.
    """
    in_ndim = x.ndim
    in_shape = x.shape

    out_shape = ()
    for k in range(in_ndim):
        if k < batch_ndim:
            out_shape += (in_shape[k],)
        else:
            out_shape += (in_shape[k] + 2 * width,)
    
    if val == 0:
        out = T.zeros(out_shape)
    else:
        out = T.ones(out_shape) * val

    indices = ()
    for k in range(0, in_ndim):
        if k < batch_ndim:
            indices += (slice(None),)
        else:
            indices += (slice(width, in_shape[k] + width),)

    return T.set_subtensor(out[indices], x)