from mock import Mock
import numpy as np
import pytest
import theano
from lasagne.layers import InputLayer, standardize, get_output


class TestExpressionLayer:
    @pytest.fixture
    def ExpressionLayer(self):
        from lasagne.layers.special import ExpressionLayer
        return ExpressionLayer

    @pytest.fixture
    def input_layer(self):
        from lasagne.layers import InputLayer
        return InputLayer((2, 3, 4, 5))

    @pytest.fixture
    def input_layer_nones(self):
        from lasagne.layers import InputLayer
        return InputLayer((1, None, None, 5))

    def np_result(self, func, input_layer):
        X = np.random.uniform(-1, 1, input_layer.output_shape)
        return X, func(X)

    @pytest.mark.parametrize('func',
                             [lambda X: X**2,
                              lambda X: X.mean(-1),
                              lambda X: X.sum(),
                              ])
    def test_tuple_shape(self, func, input_layer, ExpressionLayer):
        from lasagne.layers.helper import get_output

        X, expected = self.np_result(func, input_layer)
        layer = ExpressionLayer(input_layer, func, output_shape=expected.shape)
        assert layer.get_output_shape_for(X.shape) == expected.shape

        output = get_output(layer, X).eval()
        assert np.allclose(output, expected)

    @pytest.mark.parametrize('func',
                             [lambda X: X**2,
                              lambda X: X.mean(-1),
                              lambda X: X.sum(),
                              ])
    def test_callable_shape(self, func, input_layer, ExpressionLayer):
        from lasagne.layers.helper import get_output

        X, expected = self.np_result(func, input_layer)

        def get_shape(input_shape):
            return func(np.empty(shape=input_shape)).shape

        layer = ExpressionLayer(input_layer, func, output_shape=get_shape)
        assert layer.get_output_shape_for(X.shape) == expected.shape

        output = get_output(layer, X).eval()
        assert np.allclose(output, expected)

    @pytest.mark.parametrize('func',
                             [lambda X: X**2,
                              lambda X: X.mean(-1),
                              lambda X: X.sum(),
                              ])
    def test_none_shape(self, func, input_layer, ExpressionLayer):
        from lasagne.layers.helper import get_output

        X, expected = self.np_result(func, input_layer)

        layer = ExpressionLayer(input_layer, func, output_shape=None)
        if X.shape == expected.shape:
            assert layer.get_output_shape_for(X.shape) == expected.shape

        output = get_output(layer, X).eval()
        assert np.allclose(output, expected)

    @pytest.mark.parametrize('func',
                             [lambda X: X**2,
                              lambda X: X.mean(-1),
                              lambda X: X.sum(),
                              ])
    def test_auto_shape(self, func, input_layer, ExpressionLayer):
        from lasagne.layers.helper import get_output

        X, expected = self.np_result(func, input_layer)

        layer = ExpressionLayer(input_layer, func, output_shape='auto')
        assert layer.get_output_shape_for(X.shape) == expected.shape

        output = get_output(layer, X).eval()
        assert np.allclose(output, expected)

    @pytest.mark.parametrize('func',
                             [lambda X: X**2,
                              lambda X: X.mean(-1),
                              lambda X: X.sum(),
                              ])
    def test_nones_shape(self, func, input_layer_nones, ExpressionLayer):
        input_shape = input_layer_nones.output_shape
        np_shape = tuple(0 if s is None else s for s in input_shape)
        X = np.random.uniform(-1, 1, np_shape)
        expected = func(X)
        expected_shape = tuple(s if s else None for s in expected.shape)

        layer = ExpressionLayer(input_layer_nones,
                                func,
                                output_shape=expected_shape)
        assert layer.get_output_shape_for(input_shape) == expected_shape

        def get_shape(input_shape):
            return expected_shape
        layer = ExpressionLayer(input_layer_nones,
                                func,
                                output_shape=get_shape)
        assert layer.get_output_shape_for(input_shape) == expected_shape

        layer = ExpressionLayer(input_layer_nones,
                                func,
                                output_shape='auto')
        assert layer.get_output_shape_for(input_shape) == expected_shape


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


class TestScaleLayer:
    @pytest.fixture
    def ScaleLayer(self):
        from lasagne.layers.special import ScaleLayer
        return ScaleLayer

    @pytest.fixture
    def init_scales(self):
        # initializer for a tensor of unique values
        return lambda shape: np.arange(np.prod(shape)).reshape(shape)

    def test_scales_init(self, ScaleLayer, init_scales):
        input_shape = (2, 3, 4)
        # default: share scales over all but second axis
        b = ScaleLayer(input_shape, scales=init_scales).scales
        assert np.allclose(b.get_value(), init_scales((3,)))
        # share over first axis only
        b = ScaleLayer(input_shape, scales=init_scales, shared_axes=0).scales
        assert np.allclose(b.get_value(), init_scales((3, 4)))
        # share over second and third axis
        b = ScaleLayer(
            input_shape, scales=init_scales, shared_axes=(1, 2)).scales
        assert np.allclose(b.get_value(), init_scales((2,)))

    def test_get_output_for(self, ScaleLayer, init_scales):
        input_shape = (2, 3, 4)
        # random input tensor
        input = np.random.randn(*input_shape).astype(theano.config.floatX)
        # default: share scales over all but second axis
        layer = ScaleLayer(input_shape, scales=init_scales)
        assert np.allclose(layer.get_output_for(input).eval(),
                           input * init_scales((1, 3, 1)))
        # share over first axis only
        layer = ScaleLayer(input_shape, scales=init_scales, shared_axes=0)
        assert np.allclose(layer.get_output_for(input).eval(),
                           input * init_scales((1, 3, 4)))
        # share over second and third axis
        layer = ScaleLayer(input_shape, scales=init_scales, shared_axes=(1, 2))
        assert np.allclose(layer.get_output_for(input).eval(),
                           input * init_scales((2, 1, 1)))

    def test_undefined_shape(self, ScaleLayer):
        # should work:
        ScaleLayer((64, None, 3), shared_axes=(1, 2))
        # should not work:
        with pytest.raises(ValueError) as exc:
            ScaleLayer((64, None, 3), shared_axes=(0, 2))
        assert 'needs specified input sizes' in exc.value.args[0]


def test_standardize():
    # Simple example
    X = np.random.standard_normal((1000, 20)).astype(theano.config.floatX)
    l_in = InputLayer((None, 20))
    l_std = standardize(
        l_in, X.min(axis=0), (X.max(axis=0) - X.min(axis=0)), shared_axes=0)
    out = get_output(l_std).eval({l_in.input_var: X})
    assert np.allclose(out.max(axis=0), 1.)
    assert np.allclose(out.min(axis=0), 0.)
    # More complicated example
    X = np.random.standard_normal(
        (50, 3, 100, 10)).astype(theano.config.floatX)
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2))
    l_in = InputLayer((None, 3, None, 10))
    l_std = standardize(l_in, mean, std, shared_axes=(0, 2))
    out = get_output(l_std).eval({l_in.input_var: X})
    assert np.allclose(out.mean(axis=(0, 2)), 0., atol=1e-5)
    assert np.allclose(out.std((0, 2)), 1., atol=1e-5)


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


def test_transform_identity():
    from lasagne.layers import InputLayer, TransformerLayer
    from lasagne.utils import floatX
    from theano.tensor import constant
    batchsize = 10
    l_in = InputLayer((batchsize, 3, 28, 28))
    l_loc = InputLayer((batchsize, 6))
    layer = TransformerLayer(l_in, l_loc)
    inputs = floatX(np.arange(np.prod(l_in.shape)).reshape(l_in.shape))
    thetas = floatX(np.tile([1, 0, 0, 0, 1, 0], (batchsize, 1)))
    outputs = layer.get_output_for([constant(inputs), constant(thetas)]).eval()
    np.testing.assert_allclose(inputs, outputs, rtol=1e-6)


class TestParametricRectifierLayer:
    @pytest.fixture
    def ParametricRectifierLayer(self):
        from lasagne.layers.special import ParametricRectifierLayer
        return ParametricRectifierLayer

    @pytest.fixture
    def init_alpha(self):
        # initializer for a tensor of unique values
        return lambda shape: (np.arange(np.prod(shape)).reshape(shape)) \
            / np.prod(shape)

    def test_alpha_init(self, ParametricRectifierLayer, init_alpha):
        input_shape = (None, 3, 28, 28)
        # default: alphas only over 2nd axis
        layer = ParametricRectifierLayer(input_shape, alpha=init_alpha)
        alpha = layer.alpha
        assert layer.shared_axes == (0, 2, 3)
        assert alpha.get_value().shape == (3, )
        assert np.allclose(alpha.get_value(), init_alpha((3, )))

        # scalar alpha
        layer = ParametricRectifierLayer(input_shape, alpha=init_alpha,
                                         shared_axes='all')
        alpha = layer.alpha
        assert layer.shared_axes == (0, 1, 2, 3)
        assert alpha.get_value().shape == ()
        assert np.allclose(alpha.get_value(), init_alpha((1,)))

        # alphas shared over the 1st axis
        layer = ParametricRectifierLayer(input_shape, alpha=init_alpha,
                                         shared_axes=0)
        alpha = layer.alpha
        assert layer.shared_axes == (0,)
        assert alpha.get_value().shape == (3, 28, 28)
        assert np.allclose(alpha.get_value(), init_alpha((3, 28, 28)))

        # alphas shared over the 1st and 4th axes
        layer = ParametricRectifierLayer(input_shape, alpha=init_alpha,
                                         shared_axes=(0, 3))
        alpha = layer.alpha
        assert layer.shared_axes == (0, 3)
        assert alpha.get_value().shape == (3, 28)
        assert np.allclose(alpha.get_value(), init_alpha((3, 28)))

    def test_undefined_shape(self, ParametricRectifierLayer):
        with pytest.raises(ValueError):
            ParametricRectifierLayer((None, 3, 28, 28), shared_axes=(1, 2, 3))

    def test_get_output_for(self, ParametricRectifierLayer, init_alpha):
        input_shape = (3, 3, 28, 28)
        # random input tensor
        input = np.random.randn(*input_shape).astype(theano.config.floatX)

        # default: alphas shared only along 2nd axis
        layer = ParametricRectifierLayer(input_shape, alpha=init_alpha)
        alpha_v = layer.alpha.get_value()
        expected = np.maximum(input, 0) + np.minimum(input, 0) * \
            alpha_v[None, :, None, None]
        assert np.allclose(layer.get_output_for(input).eval(), expected)

        # scalar alpha
        layer = ParametricRectifierLayer(input_shape, alpha=init_alpha,
                                         shared_axes='all')
        alpha_v = layer.alpha.get_value()
        expected = np.maximum(input, 0) + np.minimum(input, 0) * alpha_v
        assert np.allclose(layer.get_output_for(input).eval(), expected)

        # alphas shared over the 1st axis
        layer = ParametricRectifierLayer(input_shape, alpha=init_alpha,
                                         shared_axes=0)
        alpha_v = layer.alpha.get_value()
        expected = np.maximum(input, 0) + np.minimum(input, 0) * \
            alpha_v[None, :, :, :]
        assert np.allclose(layer.get_output_for(input).eval(), expected)

        # alphas shared over the 1st and 4th axes
        layer = ParametricRectifierLayer(input_shape, shared_axes=(0, 3),
                                         alpha=init_alpha)
        alpha_v = layer.alpha.get_value()
        expected = np.maximum(input, 0) + np.minimum(input, 0) * \
            alpha_v[None, :, :, None]
        assert np.allclose(layer.get_output_for(input).eval(), expected)

    def test_prelu(self, init_alpha):
        import lasagne
        input_shape = (3, 28)
        input = np.random.randn(*input_shape).astype(theano.config.floatX)

        l_in = lasagne.layers.input.InputLayer(input_shape)
        l_dense = lasagne.layers.dense.DenseLayer(l_in, num_units=100)
        l_prelu = lasagne.layers.prelu(l_dense, alpha=init_alpha)
        output = lasagne.layers.get_output(l_prelu, input)

        assert l_dense.nonlinearity == lasagne.nonlinearities.identity

        W = l_dense.W.get_value()
        b = l_dense.b.get_value()
        alpha_v = l_prelu.alpha.get_value()
        expected = np.dot(input, W) + b
        expected = np.maximum(expected, 0) + \
            np.minimum(expected, 0) * alpha_v
        assert np.allclose(output.eval(), expected)


class TestRandomizedRectifierLayer:
    @pytest.fixture
    def RandomizedRectifierLayer(self):
        from lasagne.layers.special import RandomizedRectifierLayer
        return RandomizedRectifierLayer

    def test_high_low(self, RandomizedRectifierLayer):
        with pytest.raises(ValueError):
            RandomizedRectifierLayer((None, 3, 28, 28), lower=0.9, upper=0.1)

    def test_nomod_positive(self, RandomizedRectifierLayer):
        input = np.ones((3, 3, 28, 28)).astype(theano.config.floatX)
        layer = RandomizedRectifierLayer(input.shape)
        out = layer.get_output_for(input).eval()
        assert np.allclose(out, 1.0)

    def test_low_eq_high(self, RandomizedRectifierLayer):
        input = np.ones((3, 3, 28, 28)).astype(theano.config.floatX) * -1
        layer = RandomizedRectifierLayer(input.shape, lower=0.5, upper=0.5)
        out = layer.get_output_for(input)
        assert np.allclose(out, -0.5)

    def test_deterministic(self, RandomizedRectifierLayer):
        input = np.ones((3, 3, 28, 28)).astype(theano.config.floatX) * -1
        layer = RandomizedRectifierLayer(input.shape, lower=0.4, upper=0.6)
        out = layer.get_output_for(input, deterministic=True)
        assert np.allclose(out, -0.5)

    def test_dim_None(self, RandomizedRectifierLayer):
        import lasagne
        l_in = lasagne.layers.input.InputLayer((None, 3, 28, 28))
        layer = RandomizedRectifierLayer(l_in)
        input = np.ones((3, 3, 28, 28)).astype(theano.config.floatX)
        out = layer.get_output_for(input).eval()
        assert np.allclose(out, 1.0)

    def assert_between(self, layer, input, output):
        slopes = output / input
        slopes = slopes[input < 0]
        assert slopes.min() >= layer.lower
        assert slopes.max() <= layer.upper
        assert slopes.var() > 0

    def test_get_output_for(self, RandomizedRectifierLayer):
        input_shape = (3, 3, 28, 28)

        # ensure slope never exceeds [lower,upper)
        input = np.random.randn(*input_shape).astype(theano.config.floatX)
        layer = RandomizedRectifierLayer(input_shape, shared_axes=0)
        self.assert_between(layer, input, layer.get_output_for(input).eval())

        # from here on, we want to check parameter sharing
        # this is easier to check if the input is all ones
        input = np.ones(input_shape).astype(theano.config.floatX) * -1

        # default: parameters shared along all but 2nd axis
        layer = RandomizedRectifierLayer(input_shape)
        out = layer.get_output_for(input).eval()
        assert [
                np.allclose(out.var(axis=a), 0)
                for a in range(4)
               ] == [True, False, True, True]

        # share across all axes (single slope)
        layer = RandomizedRectifierLayer(input_shape, shared_axes='all')
        out = layer.get_output_for(input).eval()
        assert [
                np.allclose(out.var(axis=a), 0)
                for a in range(4)
               ] == [True, True, True, True]

        # share across 1st axis
        layer = RandomizedRectifierLayer(input_shape, shared_axes=0)
        out = layer.get_output_for(input).eval()
        assert [
                np.allclose(out.var(axis=a), 0)
                for a in range(4)
               ] == [True, False, False, False]

        # share across 1st and 4th axes
        layer = RandomizedRectifierLayer(input_shape, shared_axes=(0, 3))
        out = layer.get_output_for(input).eval()
        assert [
                np.allclose(out.var(axis=a), 0)
                for a in range(4)
               ] == [True, False, False, True]

    def test_rrelu(self):
        import lasagne
        input_shape = (3, 28)
        input = np.random.randn(*input_shape).astype(theano.config.floatX)

        l_in = lasagne.layers.input.InputLayer(input_shape)
        l_dense = lasagne.layers.dense.DenseLayer(l_in, num_units=100)
        l_rrelu = lasagne.layers.rrelu(l_dense)
        output = lasagne.layers.get_output(l_rrelu, input)

        assert l_dense.nonlinearity == lasagne.nonlinearities.identity

        W = l_dense.W.get_value()
        b = l_dense.b.get_value()
        self.assert_between(l_rrelu, np.dot(input, W) + b, output.eval())
