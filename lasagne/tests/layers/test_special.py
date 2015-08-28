from mock import Mock
import numpy as np
import pytest
import theano


class TestNonlinearityLayer:
    @pytest.fixture
    def NonlinearityLayer(self):
        from lasagne.layers.special import NonlinearityLayer
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
        import lasagne.nonlinearities
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


class TestBiasLayer:
    @pytest.fixture
    def BiasLayer(self):
        from lasagne.layers.special import BiasLayer
        return BiasLayer

    @pytest.fixture
    def init_b(self):
        # initializer for a tensor of unique values
        return lambda shape: np.arange(np.prod(shape)).reshape(shape)

    def test_bias_init(self, BiasLayer, init_b):
        input_shape = (2, 3, 4)
        # default: share biases over all but second axis
        b = BiasLayer(input_shape, b=init_b).b
        assert np.allclose(b.get_value(), init_b((3,)))
        # share over first axis only
        b = BiasLayer(input_shape, b=init_b, shared_axes=0).b
        assert np.allclose(b.get_value(), init_b((3, 4)))
        # share over second and third axis
        b = BiasLayer(input_shape, b=init_b, shared_axes=(1, 2)).b
        assert np.allclose(b.get_value(), init_b((2,)))
        # no bias
        b = BiasLayer(input_shape, b=None).b
        assert b is None

    def test_get_output_for(self, BiasLayer, init_b):
        input_shape = (2, 3, 4)
        # random input tensor
        input = np.random.randn(*input_shape).astype(theano.config.floatX)
        # default: share biases over all but second axis
        layer = BiasLayer(input_shape, b=init_b)
        assert np.allclose(layer.get_output_for(input).eval(),
                           input + init_b((1, 3, 1)))
        # share over first axis only
        layer = BiasLayer(input_shape, b=init_b, shared_axes=0)
        assert np.allclose(layer.get_output_for(input).eval(),
                           input + init_b((1, 3, 4)))
        # share over second and third axis
        layer = BiasLayer(input_shape, b=init_b, shared_axes=(1, 2))
        assert np.allclose(layer.get_output_for(input).eval(),
                           input + init_b((2, 1, 1)))
        # no bias
        layer = BiasLayer(input_shape, b=None)
        assert layer.get_output_for(input) is input

    def test_undefined_shape(self, BiasLayer):
        # should work:
        BiasLayer((64, None, 3), shared_axes=(1, 2))
        # should not work:
        with pytest.raises(ValueError) as exc:
            BiasLayer((64, None, 3), shared_axes=(0, 2))
        assert 'needs specified input sizes' in exc.value.args[0]


class TestInverseLayer:
    @pytest.fixture
    def invlayer_vars(self):
        from lasagne.layers.dense import DenseLayer
        from lasagne.layers.input import InputLayer
        from lasagne.layers.special import InverseLayer
        from lasagne.nonlinearities import identity

        l_in = InputLayer(shape=(10, 12))

        layer = DenseLayer(
            l_in,
            num_units=3,
            b=None,
            nonlinearity=identity,
        )

        invlayer = InverseLayer(
            incoming=layer,
            layer=layer
        )

        return {
            'layer': layer,
            'invlayer': invlayer,
        }

    def test_init(self, invlayer_vars):
        layer = invlayer_vars['layer']
        invlayer = invlayer_vars['invlayer']
        # Check that the output shape of the invlayer is the same
        # as the input shape of the layer
        assert layer.input_shape == invlayer.output_shape

    def test_get_output_shape_for(self, invlayer_vars):
        invlayer = invlayer_vars['invlayer']
        assert invlayer.get_output_shape_for(
            [(34, 55, 89, 144), (5, 8, 13, 21), (1, 1, 2, 3)]) == (1, 1, 2, 3)

    def test_get_output_for(self, invlayer_vars):
        from lasagne.layers.helper import get_output
        invlayer = invlayer_vars['invlayer']
        layer = invlayer_vars['layer']
        W = layer.W.get_value()
        input = theano.shared(
            np.random.rand(*layer.input_shape))
        results = get_output(invlayer, inputs=input)

        # Check that the output of the invlayer is the output of the
        # dot product of the output of the dense layer and the
        # transposed weights
        assert np.allclose(
            results.eval(), np.dot(np.dot(input.get_value(), W), W.T))


def test_transform_errors():
    import lasagne
    with pytest.raises(ValueError):
        l_in_a = lasagne.layers.InputLayer((None, 3, 28, 28))
        l_loc_a = lasagne.layers.DenseLayer(l_in_a, num_units=5)
        l_trans = lasagne.layers.TransformerLayer(l_in_a, l_loc_a)
    with pytest.raises(ValueError):
        l_in_b = lasagne.layers.InputLayer((3, 28, 28))
        l_loc_b = lasagne.layers.DenseLayer(l_in_b, num_units=6)
        l_trans = lasagne.layers.TransformerLayer(l_in_b, l_loc_b)


def test_transform_downsample():
        import lasagne
        downsample = (0.7, 2.3)
        x = np.random.random((10, 3, 28, 28)).astype('float32')
        x_sym = theano.tensor.tensor4()

        # create transformer with fixed input size
        l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
        l_loc = lasagne.layers.DenseLayer(l_in, num_units=6)
        l_trans = lasagne.layers.TransformerLayer(l_in, l_loc,
                                                  downsample_factor=downsample)

        # check that shape propagation works
        assert l_trans.output_shape[0] is None
        assert l_trans.output_shape[1:] == (3, int(28 / .7), int(28 / 2.3))

        # check that data propagation works
        output = lasagne.layers.get_output(l_trans, x_sym)
        x_out = output.eval({x_sym: x})
        assert x_out.shape[0] == x.shape[0]
        assert x_out.shape[1:] == l_trans.output_shape[1:]

        # create transformer with variable input size
        l_in = lasagne.layers.InputLayer((None, 3, None, 28))
        l_loc = lasagne.layers.DenseLayer(
                lasagne.layers.ReshapeLayer(l_in, ([0], 3*28*28)),
                num_units=6, W=l_loc.W, b=l_loc.b)
        l_trans = lasagne.layers.TransformerLayer(l_in, l_loc,
                                                  downsample_factor=downsample)

        # check that shape propagation works
        assert l_trans.output_shape[0] is None
        assert l_trans.output_shape[1] == 3
        assert l_trans.output_shape[2] is None
        assert l_trans.output_shape[3] == int(28 / 2.3)

        # check that data propagation works
        output = lasagne.layers.get_output(l_trans, x_sym)
        x_out2 = output.eval({x_sym: x})
        assert x_out2.shape == x_out.shape
        np.testing.assert_allclose(x_out2, x_out, rtol=1e-5, atol=1e-5)
