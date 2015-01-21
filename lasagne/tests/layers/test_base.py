from mock import Mock
import numpy
import pytest
import theano


class TestLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import Layer
        return Layer(Mock())

    def test_get_output_shape(self, layer):
        assert layer.get_output_shape() == layer.input_layer.get_output_shape()

    def test_get_output_without_arguments(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output()
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output.return_value)
        layer.input_layer.get_output.assert_called_with(None)

    def test_get_output_passes_on_arguments_to_input_layer(self, layer):
        input, arg, kwarg = object(), object(), object()
        layer.get_output_for = Mock()

        output = layer.get_output(input, arg, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output.return_value, arg, kwarg=kwarg)
        layer.input_layer.get_output.assert_called_with(
            input, arg, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer):
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_get_output_input_is_a_mapping_no_key(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output({})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer):
        input = {layer: [[1,2,3]]}
        output = layer.get_output(input)
        assert numpy.all(output.eval() == input[layer])

    def test_create_param_numpy_bad_shape_raises_error(self, layer):
        param = numpy.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(RuntimeError):
            layer.create_param(param, (3, 2))

    def test_create_param_numpy_returns_shared(self, layer):
        param = numpy.array([[1, 2, 3], [4, 5, 6]])
        result = layer.create_param(param, (2, 3))
        assert (result.get_value() == param).all()
        assert isinstance(result, type(theano.shared(param)))
        assert (result.get_value() == param).all()

    def test_create_param_shared_returns_same(self, layer):
        param = theano.shared(numpy.array([[1, 2, 3], [4, 5, 6]]))
        result = layer.create_param(param, (2, 3))
        assert result is param

    def test_create_param_callable_returns_return_value(self, layer):
        array = numpy.array([[1, 2, 3], [4, 5, 6]])
        factory = Mock()
        factory.return_value = array
        result = layer.create_param(factory, (2, 3))
        assert (result.get_value() == array).all()
        factory.assert_called_with((2, 3))


class TestMultipleInputsLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import MultipleInputsLayer
        return MultipleInputsLayer([Mock(), Mock()])

    def test_get_output_shape(self, layer):
        layer.get_output_shape_for = Mock()
        result = layer.get_output_shape()
        assert result is layer.get_output_shape_for.return_value
        layer.get_output_shape_for.assert_called_with([
            layer.input_layers[0].get_output_shape.return_value,
            layer.input_layers[1].get_output_shape.return_value,
            ])

    def test_get_output_without_arguments(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output()
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            layer.input_layers[0].get_output.return_value,
            layer.input_layers[1].get_output.return_value,
            ])
        layer.input_layers[0].get_output.assert_called_with(None)
        layer.input_layers[1].get_output.assert_called_with(None)

    def test_get_output_passes_on_arguments_to_input_layer(self, layer):
        input, arg, kwarg = object(), object(), object()
        layer.get_output_for = Mock()

        output = layer.get_output(input, arg, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            layer.input_layers[0].get_output.return_value,
            layer.input_layers[1].get_output.return_value,
            ], arg, kwarg=kwarg)
        layer.input_layers[0].get_output.assert_called_with(
            input, arg, kwarg=kwarg)
        layer.input_layers[1].get_output.assert_called_with(
            input, arg, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer):
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_get_output_input_is_a_mapping_no_key(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output({})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer):
        input = {layer: [[1,2,3]]}
        output = layer.get_output(input)
        assert numpy.all(output.eval() == input[layer])
