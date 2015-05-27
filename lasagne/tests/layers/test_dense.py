from mock import Mock
import numpy as np
import pytest
import theano


import lasagne


class TestDenseLayer:
    @pytest.fixture
    def DenseLayer(self):
        from lasagne.layers.dense import DenseLayer
        return DenseLayer

    @pytest.fixture
    def layer_vars(self, dummy_input_layer):
        from lasagne.layers.dense import DenseLayer
        W = Mock()
        b = Mock()
        nonlinearity = Mock()

        W.return_value = np.ones((12, 3))
        b.return_value = np.ones((3,)) * 3
        layer = DenseLayer(
            dummy_input_layer,
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

    def test_init_none_nonlinearity_bias(self, DenseLayer, dummy_input_layer):
        layer = DenseLayer(
            dummy_input_layer,
            num_units=3,
            nonlinearity=None,
            b=None,
            )
        assert layer.nonlinearity == lasagne.nonlinearities.identity
        assert layer.b is None

    def test_get_params(self, layer):
        assert layer.get_params() == [layer.W, layer.b]
        assert layer.get_params(regularizable=False) == [layer.b]
        assert layer.get_params(regularizable=True) == [layer.W]
        assert layer.get_params(trainable=True) == [layer.W, layer.b]
        assert layer.get_params(trainable=False) == []
        assert layer.get_params(_nonexistent_tag=True) == []
        assert layer.get_params(_nonexistent_tag=False) == [layer.W, layer.b]

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for((5, 6, 7)) == (5, 3)

    def test_get_output_for(self, layer_vars):
        layer = layer_vars['layer']
        nonlinearity = layer_vars['nonlinearity']
        W = layer_vars['W']()
        b = layer_vars['b']()

        input = theano.shared(np.ones((2, 12)))
        result = layer.get_output_for(input)
        assert result is nonlinearity.return_value

        # Check that the input to the nonlinearity was what we expect
        # from dense layer, i.e. the dot product plus bias
        nonlinearity_arg = nonlinearity.call_args[0][0]
        assert (nonlinearity_arg.eval() ==
                np.dot(input.get_value(), W) + b).all()

    def test_get_output_for_flattens_input(self, layer_vars):
        layer = layer_vars['layer']
        nonlinearity = layer_vars['nonlinearity']
        W = layer_vars['W']()
        b = layer_vars['b']()

        input = theano.shared(np.ones((2, 3, 4)))
        result = layer.get_output_for(input)
        assert result is nonlinearity.return_value

        # Check that the input to the nonlinearity was what we expect
        # from dense layer, i.e. the dot product plus bias
        nonlinearity_arg = nonlinearity.call_args[0][0]
        assert np.allclose(nonlinearity_arg.eval(),
                           np.dot(input.get_value().reshape(2, -1), W) + b)

    def test_param_names(self, layer):
        assert layer.W.name == "W"
        assert layer.b.name == "b"

    def test_named_layer_param_names(self, DenseLayer, dummy_input_layer):
        layer = DenseLayer(
            dummy_input_layer,
            num_units=3,
            name="foo"
            )

        assert layer.W.name == "foo.W"
        assert layer.b.name == "foo.b"


class TestNonlinearityLayer:
    @pytest.fixture
    def NonlinearityLayer(self):
        from lasagne.layers.dense import NonlinearityLayer
        return NonlinearityLayer

    @pytest.fixture
    def layer_vars(self, NonlinearityLayer, dummy_input_layer):
        nonlinearity = Mock()

        layer = NonlinearityLayer(
            dummy_input_layer,
            nonlinearity=nonlinearity,
            )

        return {
            'nonlinearity': nonlinearity,
            'layer': layer,
            }

    @pytest.fixture
    def layer(self, layer_vars):
        return layer_vars['layer']

    def test_init_none_nonlinearity(self, NonlinearityLayer,
                                    dummy_input_layer):
        layer = NonlinearityLayer(
            dummy_input_layer,
            nonlinearity=None,
            )
        assert layer.nonlinearity == lasagne.nonlinearities.identity

    def test_get_output_for(self, layer_vars):
        layer = layer_vars['layer']
        nonlinearity = layer_vars['nonlinearity']

        input = theano.tensor.matrix()
        result = layer.get_output_for(input)
        nonlinearity.assert_called_with(input)
        assert result is nonlinearity.return_value


class TestNINLayer:
    @pytest.fixture
    def dummy_input_layer(self):
        from lasagne.layers.input import InputLayer
        input_layer = InputLayer((2, 3, 4, 5))
        mock = Mock(input_layer)
        mock.shape = input_layer.shape
        mock.input_var = input_layer.input_var
        mock.output_shape = input_layer.output_shape
        return mock

    @pytest.fixture
    def NINLayer(self):
        from lasagne.layers.dense import NINLayer
        return NINLayer

    @pytest.fixture
    def layer_vars(self, NINLayer, dummy_input_layer):
        W = Mock()
        b = Mock()
        nonlinearity = Mock()

        W.return_value = np.ones((3, 5))
        b.return_value = np.ones((5,))
        layer = NINLayer(
            dummy_input_layer,
            num_units=5,
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
        layer_vars['W'].assert_called_with((3, 5))
        layer_vars['b'].assert_called_with((5,))

    def test_init_none_nonlinearity_bias(self, NINLayer, dummy_input_layer):
        layer = NINLayer(
            dummy_input_layer,
            num_units=3,
            nonlinearity=None,
            b=None,
            )
        assert layer.nonlinearity == lasagne.nonlinearities.identity
        assert layer.b is None

    def test_init_untie_biases(self, NINLayer, dummy_input_layer):
        layer = NINLayer(
            dummy_input_layer,
            num_units=5,
            untie_biases=True,
            )
        assert (layer.b.shape.eval() == (5, 4, 5)).all()

    def test_get_params(self, layer):
        assert layer.get_params() == [layer.W, layer.b]
        assert layer.get_params(regularizable=False) == [layer.b]
        assert layer.get_params(regularizable=True) == [layer.W]
        assert layer.get_params(trainable=True) == [layer.W, layer.b]
        assert layer.get_params(trainable=False) == []
        assert layer.get_params(_nonexistent_tag=True) == []
        assert layer.get_params(_nonexistent_tag=False) == [layer.W, layer.b]

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for((5, 6, 7, 8)) == (5, 5, 7, 8)

    @pytest.mark.parametrize("extra_kwargs", [
        {},
        {'untie_biases': True},
        {'b': None},
    ])
    def test_get_output_for(self, dummy_input_layer, extra_kwargs):
        from lasagne.layers.dense import NINLayer
        nonlinearity = Mock()

        layer = NINLayer(
            dummy_input_layer,
            num_units=6,
            nonlinearity=nonlinearity,
            **extra_kwargs
            )

        input = theano.shared(np.random.uniform(-1, 1, (2, 3, 4, 5)))
        result = layer.get_output_for(input)
        assert result is nonlinearity.return_value

        nonlinearity_arg = nonlinearity.call_args[0][0]
        X = input.get_value()
        X = np.rollaxis(X, 1).T
        X = np.dot(X, layer.W.get_value())
        if layer.b is not None:
            if layer.untie_biases:
                X += layer.b.get_value()[:, np.newaxis].T
            else:
                X += layer.b.get_value()
        X = np.rollaxis(X.T, 0, 2)
        assert np.allclose(nonlinearity_arg.eval(), X)

    def test_param_names(self, layer):
        assert layer.W.name == "W"
        assert layer.b.name == "b"

    def test_named_layer_param_names(self, NINLayer, dummy_input_layer):
        layer = NINLayer(
            dummy_input_layer,
            num_units=3,
            name="foo"
            )

        assert layer.W.name == "foo.W"
        assert layer.b.name == "foo.b"
