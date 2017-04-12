"""
Extensions depending on the cuda-convnet wrappers from pylearn2
"""

import numpy as np

import theano
import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_contiguous


def pool_2d_mp(input, ds=(2, 2), strides=None, pool_function=T.max):
    from pylearn2.sandbox.cuda_convnet.pool import MaxPool

    if strides is None:
        strides = ds

    if pool_function != T.max:
        raise NotImplementedError("Only max-pooling is supported")

    if ds[0] != ds[1]:
        raise NotImplementedError("Only square pooling regions are supported, but ds=(%d, %d)" % ds)

    if strides[0] != strides[1]:
        raise NotImplementedError("Only using the same stride in both directions is supported, but strides=(%d, %d)" % strides)

    pool_op = MaxPool(ds=ds[0], stride=strides[0])
    return pool_op(gpu_contiguous(input.dimshuffle(1, 2, 3, 0))).dimshuffle(3, 0, 1, 2)