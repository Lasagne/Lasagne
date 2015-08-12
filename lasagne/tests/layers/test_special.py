import numpy
import pytest
import theano


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
            numpy.random.rand(*layer.input_shape))
        results = get_output(invlayer, inputs=input)

        # Check that the output of the invlayer is the output of the
        # dot product of the output of the dense layer and the
        # transposed weights
        assert numpy.allclose(
            results.eval(), numpy.dot(numpy.dot(input.get_value(), W), W.T))
