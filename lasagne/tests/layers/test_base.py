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
        assert layer.input_shape == layer.input_layer.get_output_shape()

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
        input, kwarg = object(), object()
        layer.get_output_for = Mock()

        output = layer.get_output(input, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output.return_value, kwarg=kwarg)
        layer.input_layer.get_output.assert_called_with(
            input, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer):
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_get_output_input_is_a_mapping_no_key(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output({})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer):
        input = {layer: [[1, 2, 3]]}
        output = layer.get_output(input)
        assert numpy.all(output.eval() == input[layer])

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import Layer
        return Layer((None, 20))

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layer is None
        assert layer.input_shape == (None, 20)
        assert layer.get_output_shape() == (None, 20)

    def test_layer_from_shape_invalid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        with pytest.raises(RuntimeError):
            layer.get_output()
        with pytest.raises(RuntimeError):
            layer.get_output(Mock())
        with pytest.raises(RuntimeError):
            layer.get_output({Mock(): Mock()})

    def test_layer_from_shape_valid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

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

    def test_create_param_shared_bad_ndim_raises_error(self, layer):
        param = theano.shared(numpy.array([[1, 2, 3], [4, 5, 6]]))
        with pytest.raises(RuntimeError):
            layer.create_param(param, (2, 3, 4))

    def test_create_param_callable_returns_return_value(self, layer):
        array = numpy.array([[1, 2, 3], [4, 5, 6]])
        factory = Mock()
        factory.return_value = array
        result = layer.create_param(factory, (2, 3))
        assert (result.get_value() == array).all()
        factory.assert_called_with((2, 3))

    def test_named_layer(self):
        from lasagne.layers.base import Layer
        l = Layer(Mock(), name="foo")
        assert l.name == "foo"


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
        input, kwarg = object(), object()
        layer.get_output_for = Mock()

        output = layer.get_output(input, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            layer.input_layers[0].get_output.return_value,
            layer.input_layers[1].get_output.return_value,
            ], kwarg=kwarg)
        layer.input_layers[0].get_output.assert_called_with(
            input, kwarg=kwarg)
        layer.input_layers[1].get_output.assert_called_with(
            input, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer):
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_get_output_input_is_a_mapping_no_key(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output({})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer):
        input = {layer: [[1, 2, 3]]}
        output = layer.get_output(input)
        assert numpy.all(output.eval() == input[layer])

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import MultipleInputsLayer
        return MultipleInputsLayer([(None, 20), Mock()])

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layers[0] is None
        assert layer.input_shapes[0] == (None, 20)
        shape1 = layer.input_layers[1].get_output_shape()
        assert layer.input_layers[1] is not None
        assert layer.input_shapes[1] == shape1
        layer.get_output_shape_for = Mock()
        result = layer.get_output_shape()
        assert result is layer.get_output_shape_for.return_value
        layer.get_output_shape_for.assert_called_with([
            layer.input_shapes[0],
            layer.input_layers[1].get_output_shape.return_value,
            ])

    def test_layer_from_shape_invalid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        with pytest.raises(RuntimeError):
            layer.get_output()
        with pytest.raises(RuntimeError):
            layer.get_output(Mock())
        with pytest.raises(RuntimeError):
            layer.get_output({layer.input_layers[1]: Mock()})

    def test_layer_from_shape_valid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]
