import numpy as np
import pytest
import importlib
import theano

import lasagne
from lasagne.utils import floatX, as_tuple


def convNd(input, kernel, pad, stride=1, n=None):
    """Execute a batch of a stack of N-dimensional convolutions.

    Parameters
    ----------
    input : numpy array
    kernel : numpy array
    pad : {0, 'valid', 'same', 'full'}, int or tuple of int
    stride : int or tuple of int
    n : int

    Returns
    -------
    numpy array
    """
    if n is None:
        n = input.ndim - 2
    if pad not in ['valid', 'same', 'full']:
        pad = as_tuple(pad, n, int)
        input = np.pad(input, [(p, p) for p in (0, 0) + pad], mode='constant')
        pad = 'valid'

    output = np.zeros((input.shape[0], kernel.shape[0]) +
                      tuple(i + k - 1 for i, k in zip(input.shape[2:],
                                                      kernel.shape[2:])))

    if n == 1:
        for i in range(kernel.shape[2]):
            f = kernel[:, :, i:i+1]
            c = (input[:, np.newaxis] * f).sum(axis=2)
            output[:, :,
                   i:i + input.shape[2]] += c
    elif n == 2:
        for i in range(kernel.shape[2]):
            for j in range(kernel.shape[3]):
                f = kernel[:, :, i:i+1, j:j+1]
                c = (input[:, np.newaxis] * f).sum(axis=2)
                output[:, :,
                       i:i + input.shape[2],
                       j:j + input.shape[3]] += c
    elif n == 3:
        for i in range(kernel.shape[2]):
            for j in range(kernel.shape[3]):
                for k in range(kernel.shape[4]):
                    f = kernel[:, :, i:i+1, j:j+1, k:k+1]
                    c = (input[:, np.newaxis] * f).sum(axis=2)
                    output[:, :,
                           i:i + input.shape[2],
                           j:j + input.shape[3],
                           k:k + input.shape[4]] += c
    else:
        raise NotImplementedError("convNd() only supports n in (1, 2, 3)")

    if pad == 'valid':
        trim = tuple(k - 1 for k in kernel.shape[2:])
        slices = [slice(None), slice(None)]
        slices += [slice(t, -t or None) for t in trim]
        output = output[slices]
    elif pad == 'same':
        shift = tuple((k - 1) // 2 for k in kernel.shape[2:])
        slices = [slice(None), slice(None)]
        slices += [slice(s, s + i) for s, i in zip(shift, input.shape[2:])]
        output = output[slices]

    stride = as_tuple(stride, n, int)
    if any(s > 1 for s in stride):
        slices = [slice(None), slice(None)]
        slices += [slice(None, None, s) for s in stride]
        output = output[slices]

    return output


def convNd_test_sets(n):
    def _convert(input, kernel, output, kwargs):
        return [theano.shared(floatX(input)), floatX(kernel), output, kwargs]

    extra_shape = (11, 16, 23)
    input_shape = (3, 1) + extra_shape[-n:]

    for pad in (0, 1, 2, 'full', 'same'):
        for stride in (1, 2, 3):
            for filter_size in (1, 3):
                if stride > filter_size:
                    continue
                input = np.random.random(input_shape)
                kernel = np.random.random((16, 1) + (filter_size,) * n)
                output = convNd(input, kernel, pad, stride, n=n)
                yield _convert(input, kernel, output, {'pad': pad,
                                                       'stride': stride,
                                                       'flip_filters': True,
                                                       })

    # bias-less case
    input = np.random.random(input_shape)
    kernel = np.random.random((16, 1) + (3,) * n)
    output = convNd(input, kernel, pad='valid')
    yield _convert(input, kernel, output, {'b': None, 'flip_filters': True})
    # untie_biases=True case
    yield _convert(input, kernel, output, {'untie_biases': True,
                                           'flip_filters': True})
    # pad='valid' case
    yield _convert(input, kernel, output, {'pad': 'valid',
                                           'flip_filters': True})
    # flip_filters=False case
    flip = (slice(None), slice(None)) + (slice(None, None, -1),) * n
    output = convNd(input, kernel[flip], pad='valid')
    yield _convert(input, kernel, output, {'flip_filters': False})


def conv3d_test_sets():
    return convNd_test_sets(3)


def conv2d_test_sets():
    return convNd_test_sets(2)


def conv1d_test_sets():
    return convNd_test_sets(1)


def test_conv_output_length():
    from lasagne.layers.conv import conv_output_length

    assert conv_output_length(13, 5, 3, 'valid') == 3
    assert conv_output_length(13, 5, 3, 0) == 3
    assert conv_output_length(13, 5, 3, 'full') == 6
    assert conv_output_length(13, 5, 3, 'same') == 5
    assert conv_output_length(13, 5, 3, 2) == 5

    with pytest.raises(ValueError) as exc:
        conv_output_length(13, 5, 3, '_nonexistent_mode')
    assert "Invalid pad: " in exc.value.args[0]


@pytest.fixture
def DummyInputLayer():
    def factory(shape):
        from lasagne.layers.input import InputLayer
        return InputLayer(shape)
    return factory


class TestBaseConvLayer:

    def test_infer_dimensionality(self):
        from lasagne.layers.conv import BaseConvLayer
        shape = (10, 20, 30, 40, 50, 60)
        for n in range(1, 4):
            layer = BaseConvLayer(shape[:n+2], 1, 3)
            assert layer.n == n

    def test_convolve_not_implemented(self):
        from lasagne.layers.conv import BaseConvLayer
        layer = BaseConvLayer((10, 20, 30), 1, 3)
        with pytest.raises(NotImplementedError):
            layer.convolve(theano.tensor.tensor3())

    def test_fail_on_mismatching_dimensionality(self):
        from lasagne.layers.conv import BaseConvLayer
        with pytest.raises(ValueError) as exc:
            BaseConvLayer((10, 20, 30), 1, 3, n=2)
        assert "Expected 4 input dimensions" in exc.value.args[0]
        with pytest.raises(ValueError) as exc:
            BaseConvLayer((10, 20, 30, 40), 1, 3, n=1)
        assert "Expected 3 input dimensions" in exc.value.args[0]


class TestConv1DLayer:

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv1d_test_sets()))
    def test_defaults(self, DummyInputLayer,
                      input, kernel, output, kwargs):
        b, c, w = input.shape.eval()
        input_layer = DummyInputLayer((b, c, w))
        try:
            from lasagne.layers.conv import Conv1DLayer
            layer = Conv1DLayer(
                input_layer,
                num_filters=kernel.shape[0],
                filter_size=kernel.shape[2],
                W=kernel,
                **kwargs
            )
            actual = layer.get_output_for(input).eval()
            assert actual.shape == output.shape
            assert actual.shape == layer.output_shape
            assert np.allclose(actual, output)

        except NotImplementedError:
            pass

    def test_init_none_nonlinearity_bias(self, DummyInputLayer):
        from lasagne.layers.conv import Conv1DLayer
        input_layer = DummyInputLayer((1, 2, 3))
        layer = Conv1DLayer(input_layer, num_filters=16, filter_size=(3,),
                            nonlinearity=None, b=None)
        assert layer.nonlinearity == lasagne.nonlinearities.identity
        assert layer.b is None

    def test_invalid_pad(self, DummyInputLayer):
        from lasagne.layers.conv import Conv1DLayer
        input_layer = DummyInputLayer((1, 2, 3))
        with pytest.raises(TypeError) as exc:
            layer = Conv1DLayer(input_layer, num_filters=16, filter_size=(3,),
                                pad='_nonexistent_mode')
        assert "iterable of int" in exc.value.args[0]

        with pytest.raises(NotImplementedError) as exc:
            layer = Conv1DLayer(input_layer, num_filters=16, filter_size=(4,),
                                pad='same')
        assert "requires odd filter size" in exc.value.args[0]


class TestConv2DLayerImplementations:

    @pytest.fixture(
        params=[
            ('lasagne.layers', 'Conv2DLayer'),
            ('lasagne.layers.cuda_convnet', 'Conv2DCCLayer'),
            ('lasagne.layers.corrmm', 'Conv2DMMLayer'),
            ('lasagne.layers.dnn', 'Conv2DDNNLayer'),
        ],
    )
    def Conv2DImpl(self, request):
        impl_module_name, impl_name = request.param
        try:
            mod = importlib.import_module(impl_module_name)
        except ImportError:
            pytest.skip("{} not available".format(impl_module_name))

        return getattr(mod, impl_name)

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv2d_test_sets()))
    def test_defaults(self, Conv2DImpl, DummyInputLayer,
                      input, kernel, output, kwargs):
        b, c, h, w = input.shape.eval()
        input_layer = DummyInputLayer((b, c, h, w))
        try:
            layer = Conv2DImpl(
                input_layer,
                num_filters=kernel.shape[0],
                filter_size=kernel.shape[2:],
                W=kernel,
                **kwargs
            )
            actual = layer.get_output_for(input).eval()
            assert actual.shape == output.shape
            assert actual.shape == layer.output_shape
            assert np.allclose(actual, output)

        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv2d_test_sets()))
    def test_with_nones(self, Conv2DImpl, DummyInputLayer,
                        input, kernel, output, kwargs):
        if kwargs.get('untie_biases', False):
            pytest.skip()
        b, c, h, w = input.shape.eval()
        input_layer = DummyInputLayer((None, c, None, None))
        try:
            layer = Conv2DImpl(
                input_layer,
                num_filters=kernel.shape[0],
                filter_size=kernel.shape[2:],
                W=kernel,
                **kwargs
            )
            actual = layer.get_output_for(input).eval()

            assert layer.output_shape == (None,
                                          kernel.shape[0],
                                          None,
                                          None)
            assert actual.shape == output.shape
            assert np.allclose(actual, output)

        except NotImplementedError:
            pytest.skip()

    def test_init_none_nonlinearity_bias(self, Conv2DImpl, DummyInputLayer):
        input_layer = DummyInputLayer((1, 2, 3, 3))
        layer = Conv2DImpl(input_layer, num_filters=16, filter_size=(3, 3),
                           nonlinearity=None, b=None)
        assert layer.nonlinearity == lasagne.nonlinearities.identity
        assert layer.b is None

    def test_invalid_pad(self, Conv2DImpl, DummyInputLayer):
        input_layer = DummyInputLayer((1, 2, 3, 3))
        with pytest.raises(TypeError) as exc:
            layer = Conv2DImpl(input_layer, num_filters=16, filter_size=(3, 3),
                               pad='_nonexistent_mode')
        assert "iterable of int" in exc.value.args[0]

        with pytest.raises(NotImplementedError) as exc:
            layer = Conv2DImpl(input_layer, num_filters=16, filter_size=(4, 4),
                               pad='same')
        assert "requires odd filter size" in exc.value.args[0]

    def test_get_params(self, Conv2DImpl, DummyInputLayer):
        input_layer = DummyInputLayer((128, 3, 32, 32))
        layer = Conv2DImpl(input_layer, num_filters=16, filter_size=(3, 3))
        assert layer.get_params() == [layer.W, layer.b]
        assert layer.get_params(regularizable=False) == [layer.b]
        assert layer.get_params(regularizable=True) == [layer.W]
        assert layer.get_params(trainable=True) == [layer.W, layer.b]
        assert layer.get_params(trainable=False) == []
        assert layer.get_params(_nonexistent_tag=True) == []
        assert layer.get_params(_nonexistent_tag=False) == [layer.W, layer.b]


class TestConv3DLayerImplementations:

    @pytest.fixture(
        params=[
            ('lasagne.layers.dnn', 'Conv3DDNNLayer'),
        ],
    )
    def Conv3DImpl(self, request):
        impl_module_name, impl_name = request.param
        try:
            mod = importlib.import_module(impl_module_name)
        except ImportError:
            pytest.skip("{} not available".format(impl_module_name))

        return getattr(mod, impl_name)

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv3d_test_sets()))
    def test_defaults(self, Conv3DImpl, DummyInputLayer,
                      input, kernel, output, kwargs):
        b, c, h, w, d = input.shape.eval()
        input_layer = DummyInputLayer((b, c, h, w, d))
        try:
            layer = Conv3DImpl(
                input_layer,
                num_filters=kernel.shape[0],
                filter_size=kernel.shape[2:],
                W=kernel,
                **kwargs
            )
            actual = layer.get_output_for(input).eval()
            assert actual.shape == output.shape
            assert actual.shape == layer.output_shape
            assert np.allclose(actual, output)

        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv3d_test_sets()))
    def test_with_nones(self, Conv3DImpl, DummyInputLayer,
                        input, kernel, output, kwargs):
        if kwargs.get('untie_biases', False):
            pytest.skip()
        b, c, h, w, d = input.shape.eval()
        input_layer = DummyInputLayer((None, c, None, None, None))
        try:
            layer = Conv3DImpl(
                input_layer,
                num_filters=kernel.shape[0],
                filter_size=kernel.shape[2:],
                W=kernel,
                **kwargs
            )
            actual = layer.get_output_for(input).eval()

            assert layer.output_shape == (None,
                                          kernel.shape[0],
                                          None,
                                          None,
                                          None)
            assert actual.shape == output.shape
            assert np.allclose(actual, output)

        except NotImplementedError:
            pytest.skip()

    def test_init_none_nonlinearity_bias(self, Conv3DImpl, DummyInputLayer):
        input_layer = DummyInputLayer((1, 2, 3, 3, 3))
        layer = Conv3DImpl(input_layer, num_filters=16, filter_size=(3, 3, 3),
                           nonlinearity=None, b=None)
        assert layer.nonlinearity == lasagne.nonlinearities.identity
        assert layer.b is None

    def test_invalid_pad(self, Conv3DImpl, DummyInputLayer):
        input_layer = DummyInputLayer((1, 2, 3, 3, 3))
        with pytest.raises(TypeError) as exc:
            layer = Conv3DImpl(input_layer, num_filters=16,
                               filter_size=(3, 3, 3),
                               pad='_nonexistent_mode')
        assert "iterable of int" in exc.value.args[0]

        with pytest.raises(NotImplementedError) as exc:
            layer = Conv3DImpl(input_layer, num_filters=16,
                               filter_size=(4, 4, 4),
                               pad='same')
        assert "requires odd filter size" in exc.value.args[0]

    def test_get_params(self, Conv3DImpl, DummyInputLayer):
        input_layer = DummyInputLayer((128, 3, 32, 32, 32))
        layer = Conv3DImpl(input_layer, num_filters=16, filter_size=(3, 3, 3))
        assert layer.get_params() == [layer.W, layer.b]
        assert layer.get_params(regularizable=False) == [layer.b]
        assert layer.get_params(regularizable=True) == [layer.W]
        assert layer.get_params(trainable=True) == [layer.W, layer.b]
        assert layer.get_params(trainable=False) == []
        assert layer.get_params(_nonexistent_tag=True) == []
        assert layer.get_params(_nonexistent_tag=False) == [layer.W, layer.b]


class TestConv2DDNNLayer:
    def test_import_without_gpu_or_cudnn_raises(self):
        from theano.sandbox import cuda
        if cuda.cuda_enabled and cuda.dnn.dnn_available():
            pytest.skip()
        else:
            with pytest.raises(ImportError):
                import lasagne.layers.dnn


class TestConv2DMMLayer:
    def test_import_without_gpu_raises(self):
        from theano.sandbox import cuda
        if cuda.cuda_enabled:
            pytest.skip()
        else:
            with pytest.raises(ImportError):
                import lasagne.layers.corrmm


class TestConv2DCCLayer:
    def test_import_without_gpu_raises(self):
        from theano.sandbox import cuda
        if cuda.cuda_enabled:
            pytest.skip()
        else:
            with pytest.raises(ImportError):
                import lasagne.layers.cuda_convnet

    def test_unsupported_settings(self, DummyInputLayer):
        try:
            from lasagne.layers.cuda_convnet import Conv2DCCLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")

        input_layer = DummyInputLayer((128, 3, 32, 32))

        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DCCLayer(input_layer, num_filters=16,
                                  filter_size=(3, 5))
        assert ("Conv2DCCLayer only supports square filters" in
                exc.value.args[0])

        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DCCLayer(input_layer, num_filters=16,
                                  filter_size=(3, 3), stride=(1, 2))
        assert ("Conv2DCCLayer only supports square strides" in
                exc.value.args[0])

        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DCCLayer(input_layer, num_filters=15,
                                  filter_size=(3, 3))
        assert ("Conv2DCCLayer requires num_filters to be a multiple of 16" in
                exc.value.args[0])

        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DCCLayer(input_layer, num_filters=16,
                                  filter_size=(3, 3), pad=(1, 2))
        assert ("Conv2DCCLayer only supports square padding" in
                exc.value.args[0])

        input_layer = DummyInputLayer((128, 7, 32, 32))

        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DCCLayer(input_layer, num_filters=16,
                                  filter_size=(3, 3))
        assert ("Conv2DCCLayer requires the number of input channels to be "
                "1, 2, 3 or a multiple of 4" in exc.value.args[0])

    def test_pad(self, DummyInputLayer):
        try:
            from lasagne.layers.cuda_convnet import Conv2DCCLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")

        input_layer = DummyInputLayer((128, 3, 32, 32))
        layer = Conv2DCCLayer(input_layer, num_filters=16, filter_size=(3, 3),
                              pad=(3, 3))
        assert layer.output_shape == (128, 16, 36, 36)

    def test_dimshuffle_false_shapes(self, DummyInputLayer):
        try:
            from lasagne.layers.cuda_convnet import Conv2DCCLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")

        input_layer = DummyInputLayer((4, 32, 32, 128))  # c01b instead of bc01
        layer = Conv2DCCLayer(input_layer, num_filters=16, filter_size=(3, 3),
                              dimshuffle=False)
        assert layer.W.get_value().shape == (4, 3, 3, 16)
        assert layer.b.get_value().shape == (16,)

        layer = Conv2DCCLayer(input_layer, num_filters=16, filter_size=(3, 3),
                              dimshuffle=False, untie_biases=True)
        assert layer.W.get_value().shape == (4, 3, 3, 16)
        assert layer.b.get_value().shape == (16, 30, 30)

    def test_dimshuffle_false_get_output_for(self, DummyInputLayer):
        try:
            from lasagne.layers.cuda_convnet import Conv2DCCLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")

        # this implementation is tested against FilterActs instead of
        # theano.tensor.nnet.conv.conv2d because using the latter leads to
        # numerical precision errors.
        from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
        filter_acts = FilterActs(stride=1, pad=0, partial_sum=1)

        input = theano.shared(floatX(np.random.random((4, 5, 5, 8))))
        kernel = theano.shared(floatX(np.random.random((4, 3, 3, 16))))

        input_layer = DummyInputLayer((4, 5, 5, 8))  # c01b instead of bc01
        layer = Conv2DCCLayer(input_layer, num_filters=16, filter_size=(3, 3),
                              dimshuffle=False, W=kernel, b=None,
                              nonlinearity=None)

        output = np.array(filter_acts(input, kernel).eval())

        actual = layer.get_output_for(input).eval()
        actual = np.array(actual)
        assert actual.shape == output.shape
        assert actual.shape == layer.output_shape
        assert np.allclose(actual, output)


class TestShuffleLayers:
    def test_bc01_to_c01b(self):
        from lasagne.layers.input import InputLayer
        try:
            from lasagne.layers.cuda_convnet import ShuffleBC01ToC01BLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")

        input_layer = InputLayer((1, 2, 3, 4))
        layer = ShuffleBC01ToC01BLayer(input_layer)
        assert layer.output_shape == (2, 3, 4, 1)

        input = floatX(np.random.random((1, 2, 3, 4)))
        output = input.transpose(1, 2, 3, 0)
        actual = layer.get_output_for(theano.shared(input)).eval()
        assert np.allclose(output, actual)

    def test_c01b_to_bc01(self):
        from lasagne.layers.input import InputLayer
        try:
            from lasagne.layers.cuda_convnet import ShuffleC01BToBC01Layer
        except ImportError:
            pytest.skip("cuda_convnet not available")

        input_layer = InputLayer((1, 2, 3, 4))
        layer = ShuffleC01BToBC01Layer(input_layer)
        assert layer.output_shape == (4, 1, 2, 3)

        input = floatX(np.random.random((1, 2, 3, 4)))
        output = input.transpose(3, 0, 1, 2)
        actual = layer.get_output_for(theano.shared(input)).eval()
        assert np.allclose(output, actual)
