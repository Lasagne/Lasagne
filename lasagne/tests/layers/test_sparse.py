from mock import Mock
import numpy as np
import pytest
import theano.sparse as S
import scipy.sparse as sp


class TestSparseInputDenseLayer:
    @pytest.fixture
    def layer_vars(self, dummy_input_layer):
        from lasagne.layers.sparse import SparseInputDenseLayer
        W = Mock()
        b = Mock()
        nonlinearity = Mock()

        W.return_value = np.ones((12, 3))
        b.return_value = np.ones((3,)) * 3
        layer = SparseInputDenseLayer(
            dummy_input_layer,
            num_units=3,
            W=W,
            b=b,
            nonlinearity=nonlinearity)

        return {
            'W': W, 'b': b, 'nonlinearity': nonlinearity, 'layer': layer
        }

    def test_get_output_for(self, layer_vars):
        layer = layer_vars['layer']
        nonlinearity = layer_vars['nonlinearity']
        W = layer_vars['W']()
        b = layer_vars['b']()

        A = sp.eye(2, 12, format='csr')
        input = S.shared(A)
        result = layer.get_output_for(input)
        assert result is nonlinearity.return_value

        # Check input of nonlinearity, i.e W*x + b
        nonlinearity_arg = nonlinearity.call_args[0][0]
        assert(nonlinearity_arg.eval() ==
               input.get_value().dot(W) + b).all()

    def test_wrong_dense_input_raises_value_error(self, layer_vars):
        layer = layer_vars['layer']
        wrong_input = np.ones((2, 12))
        with pytest.raises(ValueError):
            layer.get_output_for(wrong_input)


class TestSparseInputDropoutLayer:
    @pytest.fixture(params=[(100, 100), (None, 100)])
    def input_layer(self, request):
        from lasagne.layers.input import InputLayer
        return InputLayer(request.param)

    @pytest.fixture
    def layer(self, input_layer):
        from lasagne.layers.sparse import SparseInputDropoutLayer
        return SparseInputDropoutLayer(input_layer)

    @pytest.fixture
    def layer_no_rescale(self, input_layer):
        from lasagne.layers.sparse import SparseInputDropoutLayer
        return SparseInputDropoutLayer(input_layer, rescale=False)

    @pytest.fixture
    def layer_p_02(self, input_layer):
        from lasagne.layers.sparse import SparseInputDropoutLayer
        return SparseInputDropoutLayer(input_layer, p=0.2)

    def test_get_output_for_non_deterministic(self, layer):
        input = S.shared(sp.eye(100, 100, format='csr'))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert(np.unique(result_eval.data) == [0., 2.]).all()
        assert 0.0085 < result_eval.mean() < 0.0115

    def test_get_output_for_deterministic(self, layer):
        input = S.shared(sp.eye(100, 100, format='csr'))
        result = layer.get_output_for(input, deterministic=True)
        result_eval = result.eval()
        assert (result_eval == input.get_value()).data.all()

    def test_wrong_dense_input_raises_value_error(self, layer):
        wrong_input = np.ones((100, 100))
        with pytest.raises(ValueError):
            layer.get_output_for(wrong_input)

    def test_get_output_for_p_float32(self, input_layer):
        from lasagne.layers.sparse import SparseInputDropoutLayer
        layer = SparseInputDropoutLayer(input_layer, p=np.float32(0.5))
        input = S.shared(sp.eye(100, 100, format='csr', dtype=np.float32))
        assert layer.get_output_for(input).dtype == input.dtype
