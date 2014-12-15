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


class TestInputLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import InputLayer
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


class TestDenseLayer:
    @pytest.fixture
    def layer_vars(self):
        from lasagne.layers.base import DenseLayer
        input_layer = Mock()
        W = Mock()
        b = Mock()
        nonlinearity = Mock()

        input_layer.get_output_shape.return_value = (2, 3, 4)
        W.return_value = numpy.ones((12, 3))
        b.return_value = numpy.ones((3,)) * 3
        layer = DenseLayer(
            input_layer=input_layer,
            num_units=3,
            W=W,
            b=b,
            nonlinearity=nonlinearity,
            )

        return {
            'W': W,
            'b': b,
            'nonlinearity': nonlinearity,
            'layer': layer,
            }

    @pytest.fixture
    def layer(self, layer_vars):
        return layer_vars['layer']

    def test_init(self, layer_vars):
        layer = layer_vars['layer']
        assert (layer.W.get_value() == layer_vars['W'].return_value).all()
        assert (layer.b.get_value() == layer_vars['b'].return_value).all()
        layer_vars['W'].assert_called_with((12, 3))
        layer_vars['b'].assert_called_with((3,))

    def test_get_params(self, layer):
        assert layer.get_params() == [layer.W, layer.b]

    def test_get_bias_params(self, layer):
        assert layer.get_bias_params() == [layer.b]

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for((5, 6, 7)) == (5, 3)

    def test_get_output_for(self, layer_vars):
        layer = layer_vars['layer']
        nonlinearity = layer_vars['nonlinearity']
        W = layer_vars['W']()
        b = layer_vars['b']()

        input = theano.shared(numpy.ones((2, 12)))
        result = layer.get_output_for(input)
        assert result is nonlinearity.return_value

        # Check that the input to the nonlinearity was what we expect
        # from dense layer, i.e. the dot product plus bias
        nonlinearity_arg = nonlinearity.call_args[0][0]
        assert (nonlinearity_arg.eval() ==
                numpy.dot(input.get_value(), W) + b).all()

    def test_get_output_for_flattens_input(self, layer_vars):
        layer = layer_vars['layer']
        nonlinearity = layer_vars['nonlinearity']
        W = layer_vars['W']()
        b = layer_vars['b']()

        input = theano.shared(numpy.ones((2, 3, 4)))
        result = layer.get_output_for(input)
        assert result is nonlinearity.return_value

        # Check that the input to the nonlinearity was what we expect
        # from dense layer, i.e. the dot product plus bias
        nonlinearity_arg = nonlinearity.call_args[0][0]
        assert (nonlinearity_arg.eval() ==
                numpy.dot(input.get_value().reshape(2, -1), W) + b).all()


class TestDropoutLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import DropoutLayer
        return DropoutLayer(Mock())

    @pytest.fixture
    def layer_no_rescale(self):
        from lasagne.layers.base import DropoutLayer
        return DropoutLayer(Mock(), rescale=False)

    @pytest.fixture
    def layer_p_02(self):
        from lasagne.layers.base import DropoutLayer
        return DropoutLayer(Mock(), p=0.2)

    def test_get_output_for_non_deterministic(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert 0.99 < result_eval.mean() < 1.01
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
        assert 0.49 < result_eval.mean() < 0.51
        assert (numpy.unique(result_eval) == [0., 1.]).all()

    def test_get_output_for_p_02(self, layer_p_02):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer_p_02.get_output_for(input)
        result_eval = result.eval()
        assert 0.99 < result_eval.mean() < 1.01
        assert (numpy.round(numpy.unique(result_eval), 2) == [0., 1.25]).all()


class TestGaussianNoiseLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import GaussianNoiseLayer
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


class TestConcatLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import ConcatLayer
        return ConcatLayer([Mock(), Mock()], axis=1)

    def test_get_output_for(self, layer):
        inputs = [theano.shared(numpy.ones((3, 3))),
            theano.shared(numpy.ones((3, 2)))]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = numpy.hstack([input.get_value() for input in inputs])
        assert (result_eval == desired_result).all()


class TestElemwiseSumLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import ElemwiseSumLayer
        return ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, -1])

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = 2*a - b
        assert (result_eval == desired_result).all()
