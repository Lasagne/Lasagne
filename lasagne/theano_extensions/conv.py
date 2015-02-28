"""
Alternative convolution implementations for Theano
"""

import numpy as np

import theano.tensor as T


# 1D convolutions

def conv1d_sc(input, filters, image_shape=None, filter_shape=None,
              border_mode='valid', subsample=(1,)):
    """
    using conv2d with a single input channel
    """
    if border_mode != 'valid':
        raise RuntimeError("Unsupported border_mode for conv1d_sc: "
                           "%s" % border_mode)

    if image_shape is None:
        image_shape_sc = None
    else:
        # (b, c, i0) to (b, 1, c, i0)
        image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2])

    if filter_shape is None:
        filter_shape_sc = None
    else:
        filter_shape_sc = (filter_shape[0], 1, filter_shape[1],
                           filter_shape[2])

    input_sc = input.dimshuffle(0, 'x', 1, 2)
    # We need to flip the channels dimension because it will be convolved over.
    filters_sc = filters.dimshuffle(0, 'x', 1, 2)[:, :, ::-1, :]

    conved = T.nnet.conv2d(input_sc, filters_sc, image_shape=image_shape_sc,
                           filter_shape=filter_shape_sc,
                           subsample=(1, subsample[0]))
    return conved[:, :, 0, :]  # drop the unused dimension


def conv1d_mc0(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,)):
    """
    using conv2d with width == 1
    """
    if image_shape is None:
        image_shape_mc0 = None
    else:
        # (b, c, i0) to (b, c, 1, i0)
        image_shape_mc0 = (image_shape[0], image_shape[1], 1, image_shape[2])

    if filter_shape is None:
        filter_shape_mc0 = None
    else:
        filter_shape_mc0 = (filter_shape[0], filter_shape[1], 1,
                            filter_shape[2])

    input_mc0 = input.dimshuffle(0, 1, 'x', 2)
    filters_mc0 = filters.dimshuffle(0, 1, 'x', 2)

    conved = T.nnet.conv2d(
        input_mc0, filters_mc0, image_shape=image_shape_mc0,
        filter_shape=filter_shape_mc0, subsample=(1, subsample[0]),
        border_mode=border_mode)
    return conved[:, :, 0, :]  # drop the unused dimension


def conv1d_mc1(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,)):
    """
    using conv2d with height == 1
    """
    if image_shape is None:
        image_shape_mc1 = None
    else:
        # (b, c, i0) to (b, c, i0, 1)
        image_shape_mc1 = (image_shape[0], image_shape[1], image_shape[2], 1)

    if filter_shape is None:
        filter_shape_mc1 = None
    else:
        filter_shape_mc1 = (filter_shape[0], filter_shape[1],
                            filter_shape[2], 1)

    input_mc1 = input.dimshuffle(0, 1, 2, 'x')
    filters_mc1 = filters.dimshuffle(0, 1, 2, 'x')

    conved = T.nnet.conv2d(
        input_mc1, filters_mc1, image_shape=image_shape_mc1,
        filter_shape=filter_shape_mc1, subsample=(subsample[0], 1),
        border_mode=border_mode)
    return conved[:, :, :, 0]  # drop the unused dimension


def conv1d_unstrided(input, filters, image_shape, filter_shape,
                     border_mode='valid', subsample=(1,),
                     implementation=conv1d_sc):
    """
    perform a strided 1D convolution by reshaping input and filters so that the
    stride becomes 1. This function requires that the filter length is a
    multiple of the stride. It also truncates the input to have a length
    that is a multiple of the stride.
    """
    batch_size, num_input_channels, input_length = image_shape
    num_filters, num_input_channels_, filter_length = filter_shape
    stride = subsample[0]

    if filter_length % stride > 0:
        raise RuntimeError("Filter length (%d) is not a multiple of the "
                           "stride (%d)" % (filter_length, stride))
    # TODO: test if this works for border_mode='full'
    assert border_mode == 'valid'

    num_steps = filter_length // stride

    # input sizes need to be multiples of the strides,
    # truncate to correct sizes.
    truncated_length = (input_length // stride) * stride
    input_truncated = input[:, :, :truncated_length]

    r_input_shape = (batch_size, num_input_channels,
                     truncated_length // stride, stride)
    r_input = input_truncated.reshape(r_input_shape)

    # fold strides into the feature maps dimension (input)
    r_input_folded_shape = (batch_size, num_input_channels * stride,
                            truncated_length // stride)
    r_input_folded = r_input.dimshuffle(
        0, 1, 3, 2).reshape(r_input_folded_shape)

    r_filter_shape = (num_filters, num_input_channels, num_steps, stride)
    r_filters_flipped = filters[:, :, ::-1].reshape(r_filter_shape)

    # fold strides into the feature maps dimension (filters)
    r_filter_folded_shape = (num_filters, num_input_channels * stride,
                             num_steps)
    r_filters_flipped_folded = r_filters_flipped.dimshuffle(
        0, 1, 3, 2).reshape(r_filter_folded_shape)
    r_filters_folded = r_filters_flipped_folded[:, :, ::-1]  # unflip

    return implementation(r_input_folded, r_filters_folded,
                          r_input_folded_shape, r_filter_folded_shape,
                          border_mode, subsample=(1,))


def conv1d_sd(input, filters, image_shape, filter_shape, border_mode='valid',
              subsample=(1,)):
    """
    using a single dot product
    """
    if border_mode != 'valid':
        raise RuntimeError("Unsupported border_mode for conv1d_sd: "
                           "%s" % border_mode)

    batch_size, num_input_channels, input_length = image_shape
    num_filters, num_input_channels_, filter_length = filter_shape
    stride = subsample[0]

    if filter_length % stride > 0:
        raise RuntimeError("Filter length (%d) is not a multiple of the "
                           "stride (%d)" % (filter_length, stride))

    num_steps = filter_length // stride
    output_length = (input_length - filter_length + stride) // stride

    # pad the input so all the shifted dot products fit inside.
    # shape is (b, c, l)
    padded_length = ((input_length // filter_length) * filter_length +
                     (num_steps - 1) * stride)

    # at this point, it is possible that the padded_length is SMALLER than the
    # input size. so then we have to truncate first.
    truncated_length = min(input_length, padded_length)
    input_truncated = input[:, :, :truncated_length]

    input_padded_shape = (batch_size, num_input_channels, padded_length)
    input_padded = T.zeros(input_padded_shape)
    input_padded = T.set_subtensor(input_padded[:, :, :truncated_length],
                                   input_truncated)

    inputs = []
    for num in range(num_steps):
        shift = num * stride
        length = (padded_length - shift) // filter_length

        r_input_shape = (batch_size, num_input_channels, length, filter_length)
        r_input = input_padded[
            :, :, shift:length * filter_length + shift].reshape(r_input_shape)

        inputs.append(r_input)

    inputs_stacked = T.stack(*inputs)  # shape is (n, b, c, w, f)
    filters_flipped = filters[:, :, ::-1]

    r_conved = T.tensordot(inputs_stacked, filters_flipped,
                           np.asarray([[2, 4], [1, 2]]))
    # resulting shape is (n, b, w, n_filters)
    # output needs to be (b, n_filters, w * n)
    r_conved = r_conved.dimshuffle(1, 3, 2, 0)  # (b, n_filters, w, n)
    conved = r_conved.reshape((r_conved.shape[0], r_conved.shape[1],
                               r_conved.shape[2] * r_conved.shape[3]))
    # result is (b, n_f, l)

    # remove padding
    return conved[:, :, :output_length]


def conv1d_md(input, filters, image_shape, filter_shape, border_mode='valid',
              subsample=(1,)):
    """
    using multiple dot products
    """
    if border_mode != 'valid':
        raise RuntimeError("Unsupported border_mode for conv1d_md: "
                           "%s" % border_mode)

    batch_size, num_input_channels, input_length = image_shape
    num_filters, num_input_channels_, filter_length = filter_shape
    stride = subsample[0]

    if filter_length % stride > 0:
        raise RuntimeError("Filter length (%d) is not a multiple of the "
                           "stride (%d)" % (filter_length, stride))

    num_steps = filter_length // stride
    output_length = (input_length - filter_length + stride) // stride
    output_shape = (batch_size, num_filters, output_length)

    filters_flipped = filters[:, :, ::-1]

    conved = T.zeros(output_shape)

    for num in range(num_steps):
        shift = num * stride
        length = (input_length - shift) // filter_length

        if length == 0:
            # we can safely skip this product, it doesn't contribute to the
            # final convolution.
            continue

        r_input_shape = (batch_size, num_input_channels, length, filter_length)
        r_input = input[
            :, :, shift:length * filter_length + shift].reshape(r_input_shape)

        # shape (b, l, n_filters)
        r_conved = T.tensordot(r_input, filters_flipped,
                               np.asarray([[1, 3], [1, 2]]))
        r_conved = r_conved.dimshuffle(0, 2, 1)  # shape is (b, n_filters, l)
        conved = T.set_subtensor(conved[:, :, num::num_steps], r_conved)

    return conved


# TODO: conv1d_md_channelslast?

# 2D convolutions

# TODO
