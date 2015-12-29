import pytest
import numpy as np
import theano.tensor as T
import lasagne


def conv1d(input, kernel, stride=1):
    output = []
    for b in input:
        temp = []
        for c in kernel:
            temp.append(
                np.convolve(b[0, :], c[0, :], mode='valid'))
        output.append(temp)
    return np.array(output)[:, :, ::stride]


@pytest.mark.parametrize('impl', ['conv1d_sc', 'conv1d_mc0',
                                  'conv1d_mc1', 'conv1d_unstrided',
                                  'conv1d_sd', 'conv1d_md'])
@pytest.mark.parametrize('filter_flip', [True, False])
@pytest.mark.parametrize('stride', [1, 2])
def test_conv(impl, stride, filter_flip):
    import lasagne.theano_extensions.conv
    conv = getattr(lasagne.theano_extensions.conv, impl)

    X = T.tensor3()
    W = T.tensor3()
    input = lasagne.utils.floatX(np.ones((1, 1, 10)))
    kernel = lasagne.utils.floatX(np.random.uniform(-1, 1, (2, 1, 6)))

    conv_theano = conv(X, W, input.shape, kernel.shape, subsample=(stride,),
                       filter_flip=filter_flip).eval({X: input, W: kernel})

    conv_np = conv1d(input, kernel, stride)

    assert np.allclose(conv_theano, conv_np)


@pytest.mark.parametrize('impl', ['conv1d_sc', 'conv1d_mc0', 'conv1d_mc1'])
def test_conv_nones(impl):
    import lasagne.theano_extensions.conv
    conv = getattr(lasagne.theano_extensions.conv, impl)

    X = T.tensor3()
    W = T.tensor3()
    input = lasagne.utils.floatX(np.ones((1, 1, 12)))
    kernel = lasagne.utils.floatX(np.random.uniform(-1, 1, (2, 1, 3)))

    conv_theano = conv(X, W, None, None).eval({
        X: input, W: kernel
        })

    conv_np = conv1d(input, kernel)

    assert np.allclose(conv_theano, conv_np)


@pytest.mark.parametrize('impl', ['conv1d_mc0', 'conv1d_mc1'])
@pytest.mark.parametrize('pad', [1, (2,)])
def test_conv_pad(impl, pad):
    import lasagne.theano_extensions.conv
    conv = getattr(lasagne.theano_extensions.conv, impl)

    X = T.tensor3()
    W = T.tensor3()
    input = lasagne.utils.floatX(np.ones((1, 1, 12)))
    kernel = lasagne.utils.floatX(np.random.uniform(-1, 1, (2, 1, 3)))

    conv_theano = conv(X, W, input.shape, kernel.shape, border_mode=pad).eval({
        X: input, W: kernel
        })

    pad = pad[0] if isinstance(pad, tuple) else pad
    input = np.pad(input, [(0, 0), (0, 0), (pad, pad)], mode='constant')
    conv_np = conv1d(input, kernel)

    assert np.allclose(conv_theano, conv_np)


@pytest.mark.parametrize('impl', ['conv1d_sc', 'conv1d_mc0',
                                  'conv1d_mc1', 'conv1d_unstrided',
                                  'conv1d_sd', 'conv1d_md'])
def test_conv_invalid_border_mode(impl):
    import lasagne.theano_extensions.conv
    conv = getattr(lasagne.theano_extensions.conv, impl)

    X = T.tensor3()
    W = T.tensor3()

    with pytest.raises(Exception):
        conv(X, W, (1, 1, 10), (2, 1, 3), border_mode=None)


@pytest.mark.parametrize('impl', ['conv1d_unstrided', 'conv1d_sd',
                                  'conv1d_md'])
def test_conv_stride(impl):
    import lasagne.theano_extensions.conv
    conv = getattr(lasagne.theano_extensions.conv, impl)

    X = T.tensor3()
    W = T.tensor3()

    with pytest.raises(Exception):
        conv(X, W, (1, 1, 10), (2, 1, 3), subsample=(2,))


@pytest.mark.parametrize('val', [0, 7])
@pytest.mark.parametrize('batch_ndim', [1, 2])
def test_pad(batch_ndim, val, width=3):
    from lasagne.theano_extensions.padding import pad

    X = T.tensor4()
    X0 = lasagne.utils.floatX(np.ones((2, 3, 4, 5)))
    X_pad_theano = pad(X, width, val, batch_ndim).eval({X: X0})

    pads = tuple((width, width) if i >= batch_ndim else (0, 0)
                 for i, _ in enumerate(X0.shape))
    X_pad_np = np.pad(X0, pads, mode='constant', constant_values=val)

    assert (X_pad_theano == X_pad_np).all()


@pytest.mark.parametrize('batch_ndim', [1, 2])
def test_pad_width_per_axis(batch_ndim, val=0):
    from lasagne.theano_extensions.padding import pad

    width = (1, 2, 3, 4)

    X = T.tensor4()
    X0 = lasagne.utils.floatX(np.ones((2, 3, 4, 5)))
    X_pad_theano = pad(X, width[batch_ndim:], val, batch_ndim).eval({X: X0})

    pads = tuple((w, w) if i >= batch_ndim else (0, 0)
                 for i, w in enumerate(width))
    X_pad_np = np.pad(X0, pads, mode='constant', constant_values=val)

    assert (X_pad_theano == X_pad_np).all()


@pytest.mark.parametrize('batch_ndim', [1, 2])
def test_pad_width_per_border(batch_ndim, val=0):
    from lasagne.theano_extensions.padding import pad

    width = [(1, 2), (3, 4), (1, 2), (3, 4)]

    X = T.tensor4()
    X0 = lasagne.utils.floatX(np.ones((2, 3, 4, 5)))
    X_pad_theano = pad(X, width[batch_ndim:], val, batch_ndim).eval({X: X0})

    pads = tuple(w if i >= batch_ndim else (0, 0)
                 for i, w in enumerate(width))
    X_pad_np = np.pad(X0, pads, mode='constant', constant_values=val)

    assert (X_pad_theano == X_pad_np).all()
