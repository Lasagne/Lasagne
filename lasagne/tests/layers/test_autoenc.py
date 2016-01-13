import numpy as np
import theano
import pytest


class TestInverseLayer:
    @pytest.fixture
    def invlayer_vars(self):
        from lasagne.layers.dense import DenseLayer
        from lasagne.layers.input import InputLayer
        from lasagne.layers.autoenc import InverseLayer
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


def test_build_autoencoder():
    import theano.tensor as T
    from lasagne.layers.input import InputLayer
    from lasagne.layers.dense import DenseLayer
    from lasagne.layers.special import BiasLayer, NonlinearityLayer
    from lasagne.layers.autoenc import InverseLayer, build_autoencoder
    from lasagne.layers.conv import Conv2DLayer
    from lasagne.layers.noise import DropoutLayer, GaussianNoiseLayer
    from lasagne.layers.pool import MaxPool2DLayer
    from lasagne.nonlinearities import identity, sigmoid, tanh
    from lasagne.layers import get_all_layers, get_output

    input_shape = (100, 3, 28, 28)
    l_in = InputLayer(input_shape)
    l1 = Conv2DLayer(l_in, num_filters=16, filter_size=(3, 3),
                     nonlinearity=tanh)
    l2 = MaxPool2DLayer(l1, pool_size=(2, 2))
    l3 = DropoutLayer(l2, p=0.5)
    l4 = DenseLayer(l3, num_units=50,
                    nonlinearity=sigmoid)
    l5 = GaussianNoiseLayer(l4,
                            sigma=0.1)
    l6 = DenseLayer(l5, num_units=10,
                    nonlinearity=tanh)
    l6_output = get_output(l6, deterministic=True)

    l_ae, l_e = build_autoencoder(l6,
                                  nonlinearity='same',
                                  b=None)
    l_e_output = get_output(l_e, deterministic=True)

    # Check output of the encoder layer
    eq_fun = theano.function([l_in.input_var],
                             T.allclose(l6_output, l_e_output))
    X_input = np.random.rand(*input_shape).astype(theano.config.floatX)
    assert eq_fun(X_input) == 1

    # List of all layers
    autoencoder_layers = get_all_layers(l_ae)

    # Check unfolding of the autoencoder
    assert isinstance(autoencoder_layers[0], InputLayer)
    assert isinstance(autoencoder_layers[1], Conv2DLayer)
    assert autoencoder_layers[1].nonlinearity == identity
    assert autoencoder_layers[1].b is None
    # Check output shape
    assert autoencoder_layers[1].output_shape == l1.output_shape
    assert isinstance(autoencoder_layers[2], BiasLayer)
    assert np.allclose(
        autoencoder_layers[2].b.get_value(),
        np.zeros_like(autoencoder_layers[2].b.get_value()))
    assert isinstance(autoencoder_layers[3], NonlinearityLayer)
    assert autoencoder_layers[3].nonlinearity == tanh
    assert isinstance(autoencoder_layers[4], MaxPool2DLayer)
    # Check output shape
    assert autoencoder_layers[4].output_shape == l2.output_shape
    assert isinstance(autoencoder_layers[5], DropoutLayer)
    assert autoencoder_layers[5].p == 0.5
    assert isinstance(autoencoder_layers[6], DenseLayer)
    assert autoencoder_layers[6].nonlinearity == identity
    assert autoencoder_layers[6].b is None
    # Check output shape
    assert autoencoder_layers[6].output_shape == l4.output_shape
    assert isinstance(autoencoder_layers[7], BiasLayer)
    assert np.allclose(
        autoencoder_layers[7].b.get_value(),
        np.zeros_like(autoencoder_layers[7].b.get_value()))
    assert isinstance(autoencoder_layers[8], NonlinearityLayer)
    assert autoencoder_layers[8].nonlinearity == sigmoid
    assert isinstance(autoencoder_layers[9], GaussianNoiseLayer)
    assert autoencoder_layers[9].sigma == 0.1
    assert isinstance(autoencoder_layers[10], DenseLayer)
    assert autoencoder_layers[10].nonlinearity == identity
    assert autoencoder_layers[10].b is None
    # Check output shape
    assert autoencoder_layers[10].output_shape == l6.output_shape
    assert isinstance(autoencoder_layers[11], BiasLayer)
    assert np.allclose(
        autoencoder_layers[11].b.get_value(),
        np.zeros_like(autoencoder_layers[11].b.get_value()))
    assert isinstance(autoencoder_layers[12], NonlinearityLayer)
    assert autoencoder_layers[12].nonlinearity == tanh
    assert isinstance(autoencoder_layers[13], InverseLayer)
    # Check output shape
    assert autoencoder_layers[13].output_shape == l6.input_shape
    assert isinstance(autoencoder_layers[14], BiasLayer)
    # Decoder bias set to None
    assert autoencoder_layers[14].b is None
    assert isinstance(autoencoder_layers[15], NonlinearityLayer)
    # Decoder nonlinearity set to 'same'
    assert autoencoder_layers[15].nonlinearity == tanh
    assert isinstance(autoencoder_layers[16], GaussianNoiseLayer)
    assert autoencoder_layers[16].sigma == 0.1
    assert isinstance(autoencoder_layers[17], InverseLayer)
    # Check output shape
    assert autoencoder_layers[17].output_shape == l4.input_shape
    assert isinstance(autoencoder_layers[18], BiasLayer)
    # Decoder bias set to None
    assert autoencoder_layers[18].b is None
    assert isinstance(autoencoder_layers[19], NonlinearityLayer)
    # Decoder nonlinearity set to 'same'
    assert autoencoder_layers[19].nonlinearity == sigmoid
    assert isinstance(autoencoder_layers[20], DropoutLayer)
    assert autoencoder_layers[20].p == 0.5
    assert isinstance(autoencoder_layers[21], InverseLayer)
    # Check output shape
    assert autoencoder_layers[21].output_shape == l2.input_shape
    # Decoder layer for pooling encoding layer
    # does not include bias or nonlinearity
    assert isinstance(autoencoder_layers[22], InverseLayer)
    # Check output shape
    assert autoencoder_layers[22].output_shape == l1.input_shape
    assert isinstance(autoencoder_layers[23], BiasLayer)
    # Decoder bias set to None
    assert autoencoder_layers[23].b is None
    assert isinstance(autoencoder_layers[24], NonlinearityLayer)
    # Decoder nonlinearity set to 'same'
    assert autoencoder_layers[24].nonlinearity == tanh


class TestBuildAutoencoderOptions:
    @pytest.fixture
    def layer_vars(self):
        from lasagne.layers import InputLayer, DenseLayer
        from lasagne.init import Constant
        from lasagne.nonlinearities import sigmoid

        input_shape = (100, 30)
        l_in = InputLayer(input_shape)
        l1 = DenseLayer(l_in, num_units=50,
                        nonlinearity=sigmoid,
                        b=Constant(0.))
        l2 = DenseLayer(l1, num_units=10,
                        nonlinearity=sigmoid,
                        b=Constant(0.))
        return {
            'l2': l2
            }

    def check_architecture(self, layer, nonlinearity, b):
        from lasagne.layers import build_autoencoder
        from lasagne.layers import InputLayer, DenseLayer, BiasLayer
        from lasagne.layers import InverseLayer, NonlinearityLayer
        from lasagne.layers import get_all_layers

        l_ae, l_e = build_autoencoder(layer,
                                      nonlinearity=nonlinearity,
                                      b=b)
        # Check that the autoencoder was correctly built
        autoencoder_layers = get_all_layers(l_ae)

        # Check architecture of autoencoder_layers for the various
        # test_build_autoencoder_biases_and_nonlinearities_as_**
        assert isinstance(autoencoder_layers[0], InputLayer)
        assert isinstance(autoencoder_layers[1], DenseLayer)
        assert isinstance(autoencoder_layers[2], BiasLayer)
        assert isinstance(autoencoder_layers[3], NonlinearityLayer)
        assert isinstance(autoencoder_layers[4], DenseLayer)
        assert isinstance(autoencoder_layers[5], BiasLayer)
        assert isinstance(autoencoder_layers[6], NonlinearityLayer)
        assert isinstance(autoencoder_layers[7], InverseLayer)
        assert isinstance(autoencoder_layers[8], BiasLayer)
        assert isinstance(autoencoder_layers[9], NonlinearityLayer)
        assert isinstance(autoencoder_layers[10], InverseLayer)
        assert isinstance(autoencoder_layers[11], BiasLayer)
        assert isinstance(autoencoder_layers[12], NonlinearityLayer)

        return autoencoder_layers

    def test_build_ae_biases_and_nonlinearities_as_callables(self, layer_vars):
        from lasagne.init import Constant
        from lasagne.nonlinearities import identity, tanh, sigmoid

        # Check that autoencoder was correctly built
        autoencoder_layers = self.check_architecture(
            layer_vars['l2'], nonlinearity=tanh, b=Constant(1.))

        # Check nonlinearities and biases
        assert autoencoder_layers[1].nonlinearity == identity
        assert autoencoder_layers[1].b is None
        assert np.allclose(
            autoencoder_layers[2].b.get_value(),
            np.zeros_like(autoencoder_layers[2].b.get_value()))
        assert autoencoder_layers[3].nonlinearity == sigmoid
        assert autoencoder_layers[4].nonlinearity == identity
        assert autoencoder_layers[4].b is None
        assert np.allclose(
            autoencoder_layers[5].b.get_value(),
            np.zeros_like(autoencoder_layers[5].b.get_value()))
        assert autoencoder_layers[6].nonlinearity == sigmoid
        assert np.allclose(
            autoencoder_layers[8].b.get_value(),
            np.ones_like(autoencoder_layers[8].b.get_value()))
        assert autoencoder_layers[9].nonlinearity == tanh
        assert np.allclose(
            autoencoder_layers[11].b.get_value(),
            np.ones_like(autoencoder_layers[11].b.get_value()))
        assert autoencoder_layers[12].nonlinearity == tanh

    def test_build_ae_biases_and_nonlinearities_as_None(self, layer_vars):
        from lasagne.nonlinearities import identity, sigmoid

        # Check that autoencoder was correctly built
        autoencoder_layers = self.check_architecture(
            layer_vars['l2'], nonlinearity=None, b=None)

        # Check that nonlinearities and biases are correctly set
        assert autoencoder_layers[1].nonlinearity == identity
        assert autoencoder_layers[1].b is None
        assert np.allclose(
            autoencoder_layers[2].b.get_value(),
            np.zeros_like(autoencoder_layers[2].b.get_value()))
        assert autoencoder_layers[3].nonlinearity == sigmoid
        assert autoencoder_layers[4].nonlinearity == identity
        assert autoencoder_layers[4].b is None
        assert np.allclose(
            autoencoder_layers[5].b.get_value(),
            np.zeros_like(autoencoder_layers[5].b.get_value()))
        assert autoencoder_layers[6].nonlinearity == sigmoid
        assert autoencoder_layers[8].b is None
        assert autoencoder_layers[9].nonlinearity == identity
        assert autoencoder_layers[11].b is None
        assert autoencoder_layers[12].nonlinearity == identity

    def test_build_ae_biases_and_nonlinearities_as_lists(self, layer_vars):
        from lasagne.init import Constant
        from lasagne.nonlinearities import identity, tanh, sigmoid, rectify

        nonlinearities_list = [tanh, rectify]
        biases_list = [Constant(1.), Constant(2.)]

        # Check that autoencoder was correctly built
        autoencoder_layers = self.check_architecture(
            layer_vars['l2'],
            nonlinearity=nonlinearities_list,
            b=biases_list)
        # Check nonlinearities and biases
        assert autoencoder_layers[1].nonlinearity == identity
        assert autoencoder_layers[1].b is None
        assert np.allclose(
            autoencoder_layers[2].b.get_value(),
            np.zeros_like(autoencoder_layers[2].b.get_value()))
        assert autoencoder_layers[3].nonlinearity == sigmoid
        assert autoencoder_layers[4].nonlinearity == identity
        assert autoencoder_layers[4].b is None
        assert np.allclose(
            autoencoder_layers[5].b.get_value(),
            np.zeros_like(autoencoder_layers[5].b.get_value()))
        assert autoencoder_layers[6].nonlinearity == sigmoid
        assert np.allclose(
            autoencoder_layers[8].b.get_value(),
            np.ones_like(autoencoder_layers[8].b.get_value()))
        assert autoencoder_layers[9].nonlinearity == tanh
        assert np.allclose(
            autoencoder_layers[11].b.get_value(),
            2.*np.ones_like(autoencoder_layers[11].b.get_value()))
        assert autoencoder_layers[12].nonlinearity == rectify


def test_unfold_bias_and_nonlinearity_layers():
    import theano.tensor as T
    from lasagne.layers import InputLayer, DenseLayer
    from lasagne.layers import BiasLayer, NonlinearityLayer
    from lasagne.nonlinearities import tanh, sigmoid, identity
    from lasagne.layers import get_all_layers, get_output
    from lasagne.layers import get_all_param_values
    from lasagne.layers import unfold_bias_and_nonlinearity_layers

    input_shape = tuple(np.random.randint(20, high=100, size=2))
    l_in = InputLayer(input_shape)
    l1 = DenseLayer(l_in, num_units=50, nonlinearity=tanh)
    l_out = DenseLayer(l1, num_units=10, nonlinearity=sigmoid)
    params_before = get_all_param_values(l_out)
    output_before = get_output(l_out)

    # add BiasLayers and NonlinearityLayers
    l_out = unfold_bias_and_nonlinearity_layers(l_out)
    params_after = get_all_param_values(l_out)
    output_after = get_output(l_out)

    # Check that all params are the same
    for p_b, p_a in zip(params_before, params_after):
        assert np.allclose(p_b, p_a)

    # Check that all layers are the right type,
    # and that all nonlinearities and biases
    # are correct
    all_layers = get_all_layers(l_out)
    assert isinstance(all_layers[0], InputLayer)
    assert isinstance(all_layers[1], DenseLayer)
    assert all_layers[1].nonlinearity == identity
    assert all_layers[1].b is None
    assert len(all_layers[1].params) == 1
    assert isinstance(all_layers[2], BiasLayer)
    assert isinstance(all_layers[3], NonlinearityLayer)
    assert all_layers[3].nonlinearity == tanh
    assert isinstance(all_layers[4], DenseLayer)
    assert all_layers[4].nonlinearity == identity
    assert all_layers[4].b is None
    assert len(all_layers[4].params) == 1
    assert isinstance(all_layers[5], BiasLayer)
    assert isinstance(all_layers[6], NonlinearityLayer)
    assert all_layers[6].nonlinearity == sigmoid

    # Check that the output before is the same as the output after
    eq_fun = theano.function([l_in.input_var],
                             T.allclose(output_before,
                                        output_after))

    X_input = np.random.rand(*input_shape).astype(theano.config.floatX)

    assert eq_fun(X_input) == 1
