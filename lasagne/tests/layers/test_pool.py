from mock import Mock
import numpy
import pytest
import theano


class TestMaxPool1DLayer:
    @pytest.fixture(params=[(32, 64, 128), (None, 64, 128)])
    def input_layer(self, request):
        return Mock(get_output_shape=lambda: request.param)

    @pytest.fixture
    def layer(self, input_layer):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(input_layer, ds=2)

    def test_get_output_for(self, layer):
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

    def test_get_output_for(self, layer):
        input = theano.shared(numpy.ones((32, 64, 24, 24)))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert result_eval.shape == (32, 64, 12, 12)

    def test_get_output_shape_for(self, layer):
        assert (layer.get_output_shape_for((None, 64, 24, 24)) ==
                (None, 64, 12, 12))
        assert (layer.get_output_shape_for((32, 64, 24, 24)) ==
                (32, 64, 12, 12))
