from mock import Mock
import numpy
import pytest
import theano


class TestLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import Layer
        return Layer(Mock())

    def test_input_shape(self, layer):
        assert layer.input_shape == layer.input_layer.output_shape

    def test_get_output_shape_for(self, layer):
        shape = Mock()
        assert layer.get_output_shape_for(shape) == shape

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import Layer
        return Layer((None, 20))

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layer is None
        assert layer.input_shape == (None, 20)

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

    def test_create_param_shared_bad_ndim_raises_error(self, layer):
        param = theano.shared(numpy.array([[1, 2, 3], [4, 5, 6]]))
        with pytest.raises(RuntimeError):
            layer.create_param(param, (2, 3, 4))

    def test_create_param_callable_returns_return_value(self, layer):
        array = numpy.array([[1, 2, 3], [4, 5, 6]])
        factory = Mock()
        factory.return_value = array
        result = layer.create_param(factory, (2, 3))
        assert (result.get_value() == array).all()
        factory.assert_called_with((2, 3))

    def test_named_layer(self):
        from lasagne.layers.base import Layer
        l = Layer(Mock(), name="foo")
        assert l.name == "foo"


class TestMergeLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import MergeLayer
        return MergeLayer([Mock(), Mock()])

    def test_input_shapes(self, layer):
        assert layer.input_shapes == [l.output_shape
                                      for l in layer.input_layers]

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.input import InputLayer
        from lasagne.layers.base import MergeLayer
        return MergeLayer([(None, 20), Mock(InputLayer((None,)))])

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layers[0] is None
        assert layer.input_shapes[0] == (None, 20)
        assert layer.input_layers[1] is not None
        assert (layer.input_shapes[1] == layer.input_layers[1].output_shape)
