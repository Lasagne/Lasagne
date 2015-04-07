from mock import Mock
import numpy as np
import pytest
import theano


def max_pool_1d(data, ds, stride=None, pad=0):
    stride = ds if stride is None else stride

    data = np.pad(data, pad, mode='constant')

    data_shifted = np.zeros((ds,) + data.shape)
    data_shifted = data_shifted[..., :data.shape[-1] - ds + 1]
    for i in range(ds):
        data_shifted[i] = data[..., i:i + data.shape[-1] - ds + 1]

    data_pooled = data_shifted.max(axis=0)

    if stride:
        data_pooled = data_pooled[..., ::stride]

    return data_pooled


def max_pool_2d(data, ds):
    data_pooled = max_pool_1d(data, ds[1])

    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = max_pool_1d(data_pooled, ds[0])
    data_pooled = np.swapaxes(data_pooled, -1, -2)

    return data_pooled


class TestMaxPool1DLayer:
    @pytest.fixture(params=[(32, 64, 128), (None, 64, 128)])
    def input_layer(self, request):
        return Mock(get_output_shape=lambda: request.param)

    @pytest.fixture
    def layer(self, input_layer):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(input_layer, ds=2)

    @pytest.fixture
    def layer_ignoreborder(self, input_layer):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(input_layer, ds=2, ignore_border=True)

    def test_get_output_for(self, layer_ignoreborder):
        input = np.random.randn(32, 64, 128)
        input_theano = theano.shared(input)
        result = layer_ignoreborder.get_output_for(input_theano)
        result_eval = result.eval()
        assert np.allclose(result_eval, max_pool_1d(input, 2))

    def test_get_output_for_shape(self, layer):
        input = theano.shared(np.ones((32, 64, 128)))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert result_eval.shape == (32, 64, 64)

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for((None, 64, 128)) == (None, 64, 64)
        assert layer.get_output_shape_for((32, 64, 128)) == (32, 64, 64)


class TestMaxPool2DLayer:
    @pytest.fixture(params=[(32, 64, 24, 24), (None, 64, 24, 24)])
    def input_layer(self, request):
        return Mock(get_output_shape=lambda: request.param)

    @pytest.fixture
    def layer(self, input_layer):
        from lasagne.layers.pool import MaxPool2DLayer
        return MaxPool2DLayer(input_layer, ds=(2, 2))

    @pytest.fixture
    def layer_ignoreborder(self, input_layer):
        from lasagne.layers.pool import MaxPool2DLayer
        return MaxPool2DLayer(input_layer, ds=(2, 2), ignore_border=True)

    def test_get_output_for(self, layer_ignoreborder):
        input = np.random.randn(32, 64, 24, 24)
        input_theano = theano.shared(input)
        result = layer_ignoreborder.get_output_for(input_theano)
        result_eval = result.eval()
        assert np.all(result_eval == max_pool_2d(input, (2, 2)))

    def test_get_output_for_shape(self, layer):
        input = theano.shared(np.ones((32, 64, 24, 24)))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert result_eval.shape == (32, 64, 12, 12)

    def test_get_output_shape_for(self, layer):
        assert (layer.get_output_shape_for((None, 64, 24, 24)) ==
                (None, 64, 12, 12))
        assert (layer.get_output_shape_for((32, 64, 24, 24)) ==
                (32, 64, 12, 12))
