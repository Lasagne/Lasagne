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

    def test_get_output_shape(self, layer):
        assert layer.get_output_shape() == (3, 2)

    def test_get_output_without_arguments(self, layer):
        assert layer.get_output() is layer.input_var

    def test_get_output_input_is_variable(self, layer):
        variable = theano.Variable("myvariable")
        assert layer.get_output(variable) is variable

    def test_get_output_input_is_array(self, layer):
        input = [[1,2,3]]
        output = layer.get_output(input)
        assert numpy.all(output.eval() == input)

    def test_get_output_input_is_a_mapping(self, layer):
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]
