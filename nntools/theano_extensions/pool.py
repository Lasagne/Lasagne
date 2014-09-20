"""
Various pooling implementations for Theano
"""

import numpy as np

import theano
import theano.tensor as T

from theano.sandbox.neighbours import images2neibs, neibs2images
from theano.tensor.signal import downsample


def pool_2d_mp(input, ds=(2, 2), strides=None, pool_function=T.max, ignore_border=False):
    if strides is None:
        strides = ds

    if pool_function != T.max:
        raise NotImplementedError("Only max-pooling is implemented, other pooling functions are not supported")

    if strides[0] != ds[0] or strides[1] != ds[1]:
        raise NotImplementedError("Only non-overlapping pooling is implemented, ds and strides do not match")

    return downsample.max_pool_2d(input, ds, ignore_border)


def pool_2d_i2n(input, ds=(2, 2), strides=None, pool_function=T.max, mode='ignore_borders'):
    if strides is None:
        strides = ds

    if strides[0] > ds[0] or strides[1] > ds[1]:
        raise RuntimeError("strides should be smaller than or equal to ds, strides=(%d, %d) and ds=(%d, %d)" %
            (strides + ds))
    
    shape = input.shape
    neibs = images2neibs(input, ds, strides, mode=mode)
    pooled_neibs = pool_function(neibs, axis=1)

    output_width = (shape[2] - ds[0]) // strides[0] + 1
    output_height = (shape[3] - ds[1]) // strides[1] + 1

    pooled_output = pooled_neibs.reshape((shape[0], shape[1], output_width, output_height))
    return pooled_output


def pool_2d_subtensor(input, ds=(2, 2), strides=None, pool_function=T.max, pad=True):
    """
    set pad to False if the input is guaranteed to have the correct input shape, i.e.
    (width - ds[0]) / strides[0] is a whole number, and (height - ds[1]) / strides[1]
    is also a whole number. This avoids a copy operation.

    This is very slow, use cuda-convnet's MaxPool instead!
    """
    if strides is None:
        strides = ds

    if strides[0] > ds[0] or strides[1] > ds[1]:
        raise RuntimeError("strides should be smaller than or equal to ds, strides=(%d, %d) and ds=(%d, %d)" %
            (strides + ds))

    if pad:
        padded_width = T.cast(T.ceil((input.shape[2] - ds[0]) / float(strides[0])) * float(strides[0]) + ds[0], 'int32')
        padded_height = T.cast(T.ceil((input.shape[3] - ds[1]) / float(strides[1])) * float(strides[1]) + ds[1], 'int32')

        offset_width = (padded_width - input.shape[2]) // 2
        offset_height = (padded_height - input.shape[3]) // 2

        padded_input = T.zeros((input.shape[0], input.shape[1], padded_width, padded_height))
        input = T.set_subtensor(padded_input[:, :, offset_width:offset_width + input.shape[2], offset_height:offset_height + input.shape[3]], input)

    parts = []
    for offset_x in range(ds[0]):
        for offset_y in range(ds[1]):
            part = input[:, :, offset_x:input.shape[2] - ds[0] + 1 + offset_x:strides[0], offset_y:input.shape[3] - ds[1] + 1 + offset_y:strides[1]]
            parts.append(part)

    stacked_parts = T.stack(*parts)
    return pool_function(stacked_parts, axis=0)


# TODO: pool_1d_mp
# TODO: pool_1d_i2n


def pool_1d_subtensor(input, ds=2, stride=None, pool_function=T.max, pad=True):
    """
    set pad to False if the input is guaranteed to have the correct input shape, i.e.
    (length - ds) / stride is a whole number. This avoids a copy operation.
    """
    if stride is None:
        stride = ds

    if stride > ds:
        raise RuntimeError("stride should be smaller than or equal to ds, stride=%d and ds=%d" %
            (stride, ds))

    if pad:
        padded_length = T.cast(T.ceil((input.shape[2] - ds) / float(stride)) * float(stride) + ds, 'int32')
        offset_length = (padded_length - input.shape[2]) // 2
        padded_input = T.zeros((input.shape[0], input.shape[1], padded_length))
        input = T.set_subtensor(padded_input[:, :, offset_length:offset_length + input.shape[2]], input)

    parts = []
    for offset in range(ds):
        part = input[:, :, offset:input.shape[2] - ds + 1 + offset:stride]
        parts.append(part)

    stacked_parts = T.stack(*parts)
    return pool_function(stacked_parts, axis=0)