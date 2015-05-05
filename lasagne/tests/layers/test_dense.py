from mock import Mock
import numpy
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

        W.return_value = numpy.ones((12, 3))
        b.return_value = numpy.ones((3,)) * 3
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

    def test_init_none_nonlinearity(self, DenseLayer, dummy_input_layer):
        layer = DenseLayer(
            dummy_input_layer,
            num_units=3,
            nonlinearity=None,
            )
        assert layer.nonlinearity == lasagne.nonlinearities.identity

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
