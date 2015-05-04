from mock import Mock
import numpy
import pytest
import theano


class TestLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import Layer
        return Layer(Mock())

    def test_input_shape(self, layer):
        assert layer.input_shape == layer.input_layer.output_shape

    def test_get_output_shape_for(self, layer):
        shape = Mock()
        assert layer.get_output_shape_for(shape) == shape

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import Layer
        return Layer((None, 20))

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layer is None
        assert layer.input_shape == (None, 20)

    def test_named_layer(self):
        from lasagne.layers.base import Layer
        l = Layer(Mock(), name="foo")
        assert l.name == "foo"


class TestMergeLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import MergeLayer
        return MergeLayer([Mock(), Mock()])

    def test_input_shapes(self, layer):
        assert layer.input_shapes == [l.output_shape
                                      for l in layer.input_layers]

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.input import InputLayer
        from lasagne.layers.base import MergeLayer
        return MergeLayer([(None, 20), Mock(InputLayer((None,)))])

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layers[0] is None
        assert layer.input_shapes[0] == (None, 20)
        assert layer.input_layers[1] is not None
        assert (layer.input_shapes[1] == layer.input_layers[1].output_shape)
