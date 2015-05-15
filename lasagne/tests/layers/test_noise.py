from mock import Mock
import numpy
import theano
import pytest


class TestDropoutLayer:
    @pytest.fixture(params=[(100, 100), (None, 100)])
    def input_layer(self, request):
        from lasagne.layers.input import InputLayer
        return InputLayer(request.param)

    @pytest.fixture
    def layer(self, input_layer):
        from lasagne.layers.noise import DropoutLayer
        return DropoutLayer(input_layer)

    @pytest.fixture
    def layer_no_rescale(self, input_layer):
        from lasagne.layers.noise import DropoutLayer
        return DropoutLayer(input_layer, rescale=False)

    @pytest.fixture
    def layer_p_02(self, input_layer):
        from lasagne.layers.noise import DropoutLayer
        return DropoutLayer(input_layer, p=0.2)

    def test_get_output_for_non_deterministic(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert 0.9 < result_eval.mean() < 1.1
        assert (numpy.unique(result_eval) == [0., 2.]).all()

    def test_get_output_for_deterministic(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input, deterministic=True)
        result_eval = result.eval()
        assert (result_eval == input.get_value()).all()

    def test_get_output_for_no_rescale(self, layer_no_rescale):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer_no_rescale.get_output_for(input)
        result_eval = result.eval()
        assert 0.4 < result_eval.mean() < 0.6
        assert (numpy.unique(result_eval) == [0., 1.]).all()

    def test_get_output_for_p_02(self, layer_p_02):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer_p_02.get_output_for(input)
        result_eval = result.eval()
        assert 0.9 < result_eval.mean() < 1.1
        assert (numpy.round(numpy.unique(result_eval), 2) == [0., 1.25]).all()


class TestGaussianNoiseLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.noise import GaussianNoiseLayer
        return GaussianNoiseLayer(Mock())

    def test_get_output_for_non_deterministic(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input, deterministic=False)
        result_eval = result.eval()
        assert (result_eval != input.eval()).all()
        assert result_eval.mean() != 1.0
        assert numpy.round(result_eval.mean()) == 1.0

    def test_get_output_for_deterministic(self, layer):
        input = theano.shared(numpy.ones((3, 3)))
        result = layer.get_output_for(input, deterministic=True)
        result_eval = result.eval()
        assert (result_eval == input.eval()).all()
