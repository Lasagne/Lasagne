import pytest
import numpy as np
import theano.tensor as T
import lasagne


@pytest.mark.parametrize('impl', ['conv1d_sc', 'conv1d_mc0',
                                  'conv1d_mc1', 'conv1d_unstrided',
                                  'conv1d_sd', 'conv1d_md'])
@pytest.mark.parametrize('stride', [1, 2])
def test_conv(impl, stride):
    import lasagne.theano_extensions.conv
    conv = getattr(lasagne.theano_extensions.conv, impl)

    X = T.tensor3()
    W = T.tensor3()
    input = lasagne.utils.floatX(np.ones((1, 1, 10)))
    kernel = lasagne.utils.floatX(np.random.uniform(-1, 1, (2, 1, 6)))

    conv_theano = conv(X, W, input.shape, kernel.shape, subsample=(stride,)
                       ).eval({X: input, W: kernel})

    output = []
    for b in input:
        temp = []
        for c in kernel:
            temp.append(
                np.convolve(b[0, :], c[0, :], mode='valid'))
        output.append(temp)
    conv_np = np.array(output)[:, :, ::stride]

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

    output = []
    for b in input:
        temp = []
        for c in kernel:
            temp.append(
                np.convolve(b[0, :], c[0, :], mode='valid'))
        output.append(temp)
    conv_np = np.array(output)

    assert np.allclose(conv_theano, conv_np)


@pytest.mark.parametrize('impl', ['conv1d_sc', 'conv1d_mc0',
                                  'conv1d_mc1', 'conv1d_unstrided',
                                  'conv1d_sd', 'conv1d_md'])
def test_conv_border_mode(impl):
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
def test_pad(val, width=3, batch_ndim=2):
    from lasagne.theano_extensions.padding import pad

    X = T.tensor4()
    X0 = lasagne.utils.floatX(np.ones((2, 3, 4, 5)))
    X_pad_theano = pad(X, width, val, batch_ndim).eval({X: X0})

    pads = tuple((width, width) if i >= batch_ndim else (0, 0)
                 for i, _ in enumerate(X0.shape))
    X_pad_np = np.pad(X0, pads, mode='constant', constant_values=val)

    assert (X_pad_theano == X_pad_np).all()
