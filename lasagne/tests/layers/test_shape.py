from mock import Mock
import numpy
import pytest
import theano


class TestReshapeLayer:
    @pytest.fixture
    def layerclass(self):
        from lasagne.layers.shape import ReshapeLayer
        return ReshapeLayer

    @pytest.fixture
    def two_unknown(self):
        from lasagne.layers.input import InputLayer
        shape = (16, 3, None, None, 10)
        return (InputLayer(shape),
                theano.shared(numpy.ones((16, 3, 5, 7, 10))))

    def test_no_reference(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (16, 3, 5, 7, 2, 5))
        assert layer.get_output_shape() == (16, 3, 5, 7, 2, 5)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_reference_both(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (-1, [1], [2], [3], 2, 5))
        assert layer.get_output_shape() == (16, 3, None, None, 2, 5)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_reference_one(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (-1, [1], [2], 7, 2, 5))
        assert layer.get_output_shape() == (None, 3, None, 7, 2, 5)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_reference_twice(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (-1, [1], [2], [3], 2, [2]))
        assert layer.get_output_shape() == (None, 3, None, None, 2, None)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_merge_with_unknown(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, ([0], [1], [2], -1))
        assert layer.get_output_shape() == (16, 3, None, None)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 70)

    def test_merge_two_unknowns(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, ([0], [1], -1, [4]))
        assert layer.get_output_shape() == (16, 3, None, 10)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 35, 10)

    def test_size_mismatch(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (17, 3, [2], [3], -1))
        with pytest.raises(ValueError) as excinfo:
            layer.get_output_shape() == (16, 3, None, 10)
        assert 'match' in str(excinfo.value)

    def test_invalid_spec(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        with pytest.raises(ValueError):
            layerclass(inputlayer, (-16, 3, 5, 7, 10))
        with pytest.raises(ValueError):
            layerclass(inputlayer, (-1, 3, 5, 7, -1))
        with pytest.raises(ValueError):
            layerclass(inputlayer, ([-1], 3, 5, 7, 10))
        with pytest.raises(ValueError):
            layerclass(inputlayer, ([0, 1], 3, 5, 7, 10))
        with pytest.raises(ValueError):
            layerclass(inputlayer, (None, 3, 5, 7, 10))
