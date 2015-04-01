from mock import Mock
import numpy
import pytest
import theano


def max_pool_1d(data, ds):
    data_truncated = data[:, :, :(data.shape[2] // ds) * ds]
    data_pooled = data_truncated.reshape((-1, ds)).max(axis=1)
    return data_pooled.reshape(data.shape[:2] + (data.shape[2] // ds,))


def max_pool_2d(data, ds):
    data_truncated = data[:, :, :(data.shape[2] // ds[0]) * ds[0],
                          :(data.shape[3] // ds[1]) * ds[1]]
    data_reshaped = data_truncated.reshape((-1, data.shape[2] // ds[0], ds[0],
                                            data.shape[3] // ds[1], ds[1]))

    data_pooled = data_reshaped.max(axis=4).max(axis=2)

    return data_pooled.reshape(data.shape[:2] + (data.shape[2] // ds[0],
                                                 data.shape[3] // ds[1]))


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
        input = numpy.random.randn(32, 64, 128)
        input_theano = theano.shared(input)
        result = layer_ignoreborder.get_output_for(input_theano)
        result_eval = result.eval()
        assert numpy.all(result_eval == max_pool_1d(input, 2))

    def test_get_output_for_shape(self, layer):
        input = theano.shared(numpy.ones((32, 64, 128)))
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
        input = numpy.random.randn(32, 64, 24, 24)
        input_theano = theano.shared(input)
        result = layer_ignoreborder.get_output_for(input_theano)
        result_eval = result.eval()
        assert numpy.all(result_eval == max_pool_2d(input, (2, 2)))

    def test_get_output_for_shape(self, layer):
        input = theano.shared(numpy.ones((32, 64, 24, 24)))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert result_eval.shape == (32, 64, 12, 12)

    def test_get_output_shape_for(self, layer):
        assert (layer.get_output_shape_for((None, 64, 24, 24)) ==
                (None, 64, 12, 12))
        assert (layer.get_output_shape_for((32, 64, 24, 24)) ==
                (32, 64, 12, 12))
