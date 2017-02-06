import pytest
import theano


class TestInputLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.input import InputLayer
        return InputLayer((3, 2))

    def test_input_var(self, layer):
        assert layer.input_var.ndim == 2

    def test_shape(self, layer):
        assert layer.shape == (3, 2)

    def test_shape_list(self, layer):
        from lasagne.layers.input import InputLayer
        assert InputLayer([3, 2]).shape == (3, 2)

    def test_input_var_bcast(self):
        from lasagne.layers.input import InputLayer
        assert InputLayer((3, 2)).input_var.broadcastable == (False, False)
        assert InputLayer((1, 2)).input_var.broadcastable == (True, False)
        assert InputLayer((None, 1)).input_var.broadcastable == (False, True)

    def test_input_var_name(self, layer):
        assert layer.input_var.name == "input"

    def test_named_layer_input_var_name(self):
        from lasagne.layers.input import InputLayer
        layer = InputLayer((3, 2), name="foo")
        assert layer.input_var.name == "foo.input"

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_bad_shape_fails(self):
        from lasagne.layers.input import InputLayer
        input_var = theano.tensor.tensor4()

        with pytest.raises(ValueError):
            InputLayer((3, 2), input_var)

    def test_nonpositive_input_dims_raises_value_error(self):
        from lasagne.layers import InputLayer
        with pytest.raises(ValueError):
            InputLayer(shape=(None, -1, -1))
        with pytest.raises(ValueError):
            InputLayer(shape=(None, 0, 0))
        InputLayer(shape=(None, 1, 1))
