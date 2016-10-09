import pytest
import numpy as np
import theano.tensor as T
import lasagne

from collections import OrderedDict
from theano.scan_module.scan_utils import equal_computations
from mock import Mock


class TestRegularizationPenalties(object):
    def l1(self, x):
        return np.abs(x).sum()

    def l2(self, x):
        return (x**2).sum()

    @pytest.mark.parametrize('penalty',
                             ['l1', 'l2'])
    def test_penalty(self, penalty):
        np_penalty = getattr(self, penalty)
        theano_penalty = getattr(lasagne.regularization, penalty)

        X = T.matrix()
        X0 = lasagne.utils.floatX(np.random.uniform(-3, 3, (10, 10)))

        theano_result = theano_penalty(X).eval({X: X0})
        np_result = np_penalty(X0)

        assert np.allclose(theano_result, np_result)


class TestRegularizationHelpers(object):
    @pytest.fixture
    def layers(self):
        l_1 = lasagne.layers.InputLayer((None, 10))
        l_2 = lasagne.layers.DenseLayer(l_1, num_units=20)
        l_3 = lasagne.layers.DenseLayer(l_2, num_units=30)
        return l_1, l_2, l_3

    def test_apply_penalty(self):
        from lasagne.regularization import apply_penalty, l2
        A = T.vector()
        B = T.matrix()

        assert apply_penalty([], l2) == 0

        assert equal_computations([apply_penalty(A, l2)],
                                  [l2(A)])

        assert equal_computations([apply_penalty([A, B], l2)],
                                  [sum([l2(A), l2(B)])])

    def test_regularize_layer_params_single_layer(self, layers):
        from lasagne.regularization import regularize_layer_params
        l_1, l_2, l_3 = layers

        penalty = Mock(return_value=0)
        loss = regularize_layer_params(l_2, penalty)

        assert penalty.call_count == 1
        penalty.assert_any_call(l_2.W)

    def test_regularize_layer_params_multiple_layers(self, layers):
        from lasagne.regularization import regularize_layer_params
        l_1, l_2, l_3 = layers

        penalty = Mock(return_value=0)
        loss = regularize_layer_params([l_1, l_2, l_3], penalty)

        assert penalty.call_count == 2
        penalty.assert_any_call(l_2.W)
        penalty.assert_any_call(l_3.W)

    def test_regularize_network_params(self, layers):
        from lasagne.regularization import regularize_network_params
        l_1, l_2, l_3 = layers

        penalty = Mock(return_value=0)
        loss = regularize_network_params(l_3, penalty)

        assert penalty.call_count == 2
        penalty.assert_any_call(l_2.W)
        penalty.assert_any_call(l_3.W)

    def test_regularize_layer_params_weighted(self, layers):
        from lasagne.regularization import regularize_layer_params_weighted
        from lasagne.regularization import apply_penalty, l2
        l_1, l_2, l_3 = layers

        layers = OrderedDict()
        layers[l_2] = 0.1
        layers[l_3] = 0.5

        loss = regularize_layer_params_weighted(layers,
                                                lasagne.regularization.l2)
        assert equal_computations([loss],
                                  [sum([0.1 * apply_penalty([l_2.W], l2),
                                        0.5 * apply_penalty([l_3.W], l2)])])
