import numpy as np
import pytest
import importlib
import theano

import lasagne
from lasagne.utils import floatX


def conv2d(input, kernel, border_mode):
    output = np.zeros((input.shape[0],
                       kernel.shape[0],
                       input.shape[2] + kernel.shape[2] - 1,
                       input.shape[3] + kernel.shape[3] - 1,
                       ))
    for i in range(kernel.shape[2]):
        for j in range(kernel.shape[3]):
            k = kernel[:, :, i, j][:, :, np.newaxis, np.newaxis]
            output[:, :, i:i + input.shape[2],
                   j:j + input.shape[3]] += (input[:, np.newaxis] * k).sum(2)

    if border_mode == 'valid':
        trim = (kernel.shape[2] - 1, kernel.shape[3] - 1)
        output = output[:, :, trim[0]:-trim[0], trim[1]:-trim[1]]

    elif border_mode == 'same':
        shift_x = (kernel.shape[2] - 1) // 2
        shift_y = (kernel.shape[3] - 1) // 2
        output = output[:, :, shift_x:input.shape[2] + shift_x,
                        shift_y:input.shape[3] + shift_y]
    return output


def conv2d_test_sets():
    def _convert(input, kernel, output, kwargs):
        return [theano.shared(floatX(input)), floatX(kernel), output, kwargs]

    for border_mode in ['valid', 'full', 'same']:
        for stride in [1, 2, 3]:
            input = np.random.random((3, 1, 16, 23))
            kernel = np.random.random((16, 1, 3, 3))
            output = conv2d(input, kernel, border_mode=border_mode)
            output = output[:, :, ::stride, ::stride]
            yield _convert(input, kernel, output, {'border_mode': border_mode,
                                                   'stride': stride
                                                   })

            input = np.random.random((3, 3, 16, 23))
            kernel = np.random.random((16, 3, 3, 3))
            output = conv2d(input, kernel, border_mode=border_mode)
            output = output[:, :, ::stride, ::stride]
            yield _convert(input, kernel, output, {'border_mode': border_mode,
                                                   'stride': stride
                                                   })

    # bias-less case
    input = np.random.random((3, 1, 16, 23))
    kernel = np.random.random((16, 1, 3, 3))
    output = conv2d(input, kernel, border_mode='valid')
    yield _convert(input, kernel, output, {'b': None})


def conv1d(input, kernel, border_mode='valid'):
    output = []
    for b in input:
        temp = []
        for c in kernel:
            temp.append(
                np.convolve(b[0, :], c[0, :], mode=border_mode))
        output.append(temp)
    return np.array(output)


def conv1d_test_sets():
    def _convert(input, kernel, output, kwargs):
        return [theano.shared(floatX(input)), floatX(kernel), output, kwargs]

    for border_mode in ['valid', 'full', 'same']:
        for stride in [1, 2, 3]:
            input = np.random.random((3, 1, 23))
            kernel = np.random.random((16, 1, 3))
            output = conv1d(input, kernel, border_mode)
            output = output[:, :, ::stride]
            yield _convert(input, kernel, output, {'border_mode': border_mode,
                                                   'stride': stride,
                                                   })

    # bias-less case
    input = np.random.random((3, 1, 23))
    kernel = np.random.random((16, 1, 3))
    output = conv1d(input, kernel, border_mode='valid')
    yield _convert(input, kernel, output, {'b': None})


def test_conv_output_length():
    from lasagne.layers.conv import conv_output_length

    assert conv_output_length(13, 5, 3, 'valid', 2) == 3
    assert conv_output_length(13, 5, 3, 'full', 2) == 6
    assert conv_output_length(13, 5, 3, 'same', 2) == 5
    assert conv_output_length(13, 5, 3, 'pad', 2) == 5

    with pytest.raises(ValueError) as exc:
        conv_output_length(13, 5, 3, '_nonexistent_mode', 2)
    assert "Invalid border mode" in exc.value.args[0]


@pytest.fixture
def DummyInputLayer():
    def factory(shape):
        from lasagne.layers.input import InputLayer
        return InputLayer(shape)
    return factory


class TestConv1DLayer:

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv1d_test_sets()))
    @pytest.mark.parametrize("extra_kwargs", [
        {},
        {'untie_biases': True},
    ])
    def test_defaults(self, DummyInputLayer,
                      input, kernel, output, kwargs, extra_kwargs):
        kwargs.update(extra_kwargs)
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

    def test_invalid_border_mode(self, DummyInputLayer):
        from lasagne.layers.conv import Conv1DLayer
        input_layer = DummyInputLayer((1, 2, 3))
        with pytest.raises(RuntimeError) as exc:
            layer = Conv1DLayer(input_layer, num_filters=16, filter_size=(3,),
                                border_mode='_nonexistent_mode')
        assert "Invalid border mode" in exc.value.args[0]


class TestConv2DLayerImplementations:

    @pytest.fixture(
        params=[
            ('lasagne.layers', 'Conv2DLayer', {}),
            ('lasagne.layers.cuda_convnet',
             'Conv2DCCLayer',
             {'flip_filters': True}),
            ('lasagne.layers.corrmm', 'Conv2DMMLayer', {'flip_filters': True}),
            ('lasagne.layers.dnn', 'Conv2DDNNLayer', {'flip_filters': True}),
        ],
    )
    def Conv2DImpl(self, request):
        impl_module_name, impl_name, impl_default_kwargs = request.param
        try:
            mod = importlib.import_module(impl_module_name)
        except ImportError:
            pytest.skip("{} not available".format(impl_module_name))

        impl = getattr(mod, impl_name)

        def wrapper(*args, **kwargs):
            kwargs2 = impl_default_kwargs.copy()
            kwargs2.update(kwargs)
            return impl(*args, **kwargs2)

        wrapper.__name__ = impl_name
        return wrapper

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv2d_test_sets()))
    @pytest.mark.parametrize("extra_kwargs", [
        {},
        {'untie_biases': True},
    ])
    def test_defaults(self, Conv2DImpl, DummyInputLayer,
                      input, kernel, output, kwargs, extra_kwargs):
        kwargs.update(extra_kwargs)
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

    def test_invalid_border_mode(self, Conv2DImpl, DummyInputLayer):
        input_layer = DummyInputLayer((1, 2, 3))
        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DImpl(input_layer, num_filters=16, filter_size=(3, 3),
                               border_mode='_nonexistent_mode')
        assert "Invalid border mode" in exc.value.args[0]

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


class TestConv2DDNNLayer:
    def test_import_without_gpu_or_cudnn_raises(self):
        from theano.sandbox.cuda import dnn
        if theano.config.device.startswith("gpu") and dnn.dnn_available():
            pytest.skip()
        else:
            with pytest.raises(ImportError):
                import lasagne.layers.dnn

    def test_pad(self, DummyInputLayer):
        try:
            from lasagne.layers.dnn import Conv2DDNNLayer
        except ImportError:
            pytest.skip("dnn not available")

        input_layer = DummyInputLayer((1, 2, 3, 3))
        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DDNNLayer(input_layer, num_filters=1,
                                   filter_size=(3, 3), border_mode='valid',
                                   pad=(1, 1))
        assert ("You cannot specify both 'border_mode' and 'pad'" in
                exc.value.args[0])

        layer = Conv2DDNNLayer(input_layer, num_filters=4, filter_size=(3, 3),
                               pad=(3, 3))
        assert layer.output_shape == (1, 4, 7, 7)


class TestConv2DMMLayer:
    def test_import_without_gpu_raises(self):
        if theano.config.device.startswith("gpu"):
            pytest.skip()
        else:
            with pytest.raises(ImportError):
                import lasagne.layers.corrmm

    def test_pad(self, DummyInputLayer):
        try:
            from lasagne.layers.corrmm import Conv2DMMLayer
        except ImportError:
            pytest.skip("corrmm not available")

        input_layer = DummyInputLayer((1, 2, 3, 3))
        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DMMLayer(input_layer, num_filters=1,
                                  filter_size=(3, 3), border_mode='valid',
                                  pad=(1, 1))
        assert ("You cannot specify both 'border_mode' and 'pad'" in
                exc.value.args[0])

        layer = Conv2DMMLayer(input_layer, num_filters=4, filter_size=(3, 3),
                              pad=(3, 3))
        assert layer.output_shape == (1, 4, 7, 7)


class TestConv2DCCLayer:
    def test_import_without_gpu_raises(self):
        if theano.config.device.startswith("gpu"):
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
        with pytest.raises(RuntimeError) as exc:
            layer = Conv2DCCLayer(input_layer, num_filters=16,
                                  filter_size=(3, 3), border_mode='valid',
                                  pad=(1, 1))
        assert ("You cannot specify both 'border_mode' and 'pad'" in
                exc.value.args[0])

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
