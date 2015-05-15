import numpy
import pytest
import theano


class TestInputLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.input import InputLayer
        return InputLayer((3, 2))

    def test_input_var(self, layer):
        assert layer.input_var.ndim == 2

    def test_shape(self, layer):
        assert layer.shape == (3, 2)

    def test_input_var_name(self, layer):
        assert layer.input_var.name == "input"

    def test_named_layer_input_var_name(self):
        from lasagne.layers.input import InputLayer
        layer = InputLayer((3, 2), name="foo")
        assert layer.input_var.name == "foo.input"
