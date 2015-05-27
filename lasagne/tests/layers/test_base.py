from mock import Mock
import numpy
import pytest
import theano


class TestLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import Layer
        return Layer(Mock())

    @pytest.fixture
    def named_layer(self):
        from lasagne.layers.base import Layer
        return Layer(Mock(), name='layer_name')

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

    def test_named_layer(self, named_layer):
        assert named_layer.name == 'layer_name'

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_get_params_tags(self, layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = layer.add_param(a, a_shape, name='A', tag1=True, tag2=False)

        b_shape = (30, 20)
        b = numpy.random.normal(0, 1, b_shape)
        B = layer.add_param(b, b_shape, name='B', tag1=True, tag2=True)

        c_shape = (40, 10)
        c = numpy.random.normal(0, 1, c_shape)
        C = layer.add_param(c, c_shape, name='C', tag2=True)

        assert layer.get_params() == [A, B, C]
        assert layer.get_params(tag1=True) == [A, B]
        assert layer.get_params(tag1=False) == [C]
        assert layer.get_params(tag2=True) == [B, C]
        assert layer.get_params(tag2=False) == [A]
        assert layer.get_params(tag1=True, tag2=True) == [B]

    def test_add_param_tags(self, layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = layer.add_param(a, a_shape)
        assert A in layer.params
        assert 'trainable' in layer.params[A]
        assert 'regularizable' in layer.params[A]

        b_shape = (30, 20)
        b = numpy.random.normal(0, 1, b_shape)
        B = layer.add_param(b, b_shape, trainable=False)
        assert B in layer.params
        assert 'trainable' not in layer.params[B]
        assert 'regularizable' in layer.params[B]

        c_shape = (40, 10)
        c = numpy.random.normal(0, 1, c_shape)
        C = layer.add_param(c, c_shape, tag1=True)
        assert C in layer.params
        assert 'trainable' in layer.params[C]
        assert 'regularizable' in layer.params[C]
        assert 'tag1' in layer.params[C]

    def test_add_param_name(self, layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = layer.add_param(a, a_shape, name='A')
        assert A.name == 'A'

    def test_add_param_named_layer_name(self, named_layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = named_layer.add_param(a, a_shape, name='A')
        assert A.name == 'layer_name.A'

    def test_get_output_for_notimplemented(self, layer):
        with pytest.raises(NotImplementedError):
            layer.get_output_for(Mock())


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

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_get_output_shape_for_notimplemented(self, layer):
        with pytest.raises(NotImplementedError):
            layer.get_output_shape_for(Mock())

    def test_get_output_for_notimplemented(self, layer):
        with pytest.raises(NotImplementedError):
            layer.get_output_for(Mock())
