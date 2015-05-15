import numpy
import pytest
import theano


class TestReshapeLayer:
    @pytest.fixture
    def layerclass(self):
        from lasagne.layers.shape import ReshapeLayer
        return ReshapeLayer

    @pytest.fixture
    def two_unknown(self):
        from lasagne.layers.input import InputLayer
        shape = (16, 3, None, None, 10)
        return (InputLayer(shape),
                theano.shared(numpy.ones((16, 3, 5, 7, 10))))

    def test_no_reference(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (16, 3, 5, 7, 2, 5))
        assert layer.output_shape == (16, 3, 5, 7, 2, 5)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_reference_both(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (-1, [1], [2], [3], 2, 5))
        assert layer.output_shape == (16, 3, None, None, 2, 5)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_reference_one(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (-1, [1], [2], 7, 2, 5))
        assert layer.output_shape == (None, 3, None, 7, 2, 5)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_reference_twice(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, (-1, [1], [2], [3], 2, [2]))
        assert layer.output_shape == (None, 3, None, None, 2, None)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)

    def test_merge_with_unknown(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, ([0], [1], [2], -1))
        assert layer.output_shape == (16, 3, None, None)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 5, 70)

    def test_merge_two_unknowns(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        layer = layerclass(inputlayer, ([0], [1], -1, [4]))
        assert layer.output_shape == (16, 3, None, 10)
        result = layer.get_output_for(inputdata).eval()
        assert result.shape == (16, 3, 35, 10)

    def test_size_mismatch(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        with pytest.raises(ValueError) as excinfo:
            layerclass(inputlayer, (17, 3, [2], [3], -1))
        assert 'match' in str(excinfo.value)

    def test_invalid_spec(self, layerclass, two_unknown):
        inputlayer, inputdata = two_unknown
        with pytest.raises(ValueError):
            layerclass(inputlayer, (-16, 3, 5, 7, 10))
        with pytest.raises(ValueError):
            layerclass(inputlayer, (-1, 3, 5, 7, -1))
        with pytest.raises(ValueError):
            layerclass(inputlayer, ([-1], 3, 5, 7, 10))
        with pytest.raises(ValueError):
            layerclass(inputlayer, ([0, 1], 3, 5, 7, 10))
        with pytest.raises(ValueError):
            layerclass(inputlayer, (None, 3, 5, 7, 10))
        with pytest.raises(ValueError):
            layerclass(inputlayer, (16, 3, 5, 7, [5]))


class TestDimshuffleLayer:
    @pytest.fixture
    def input_shape(self):
        return (2, 3, 1, 5, 7)

    @pytest.fixture
    def input_var(self):
        InputTensorType = theano.tensor.TensorType(
            'float64', broadcastable=(False, False, True, False, False),
            name='DimShuffleTestTensor')
        return InputTensorType(name='x')

    @pytest.fixture
    def input_layer(self, input_shape, input_var):
        from lasagne.layers.input import InputLayer
        return InputLayer(input_shape, input_var)

    @pytest.fixture
    def input_shape_with_None(self):
        return (2, 3, None, 5, 7)

    @pytest.fixture
    def input_var(self):
        InputTensorType = theano.tensor.TensorType(
            'float64', broadcastable=(False, False, True, False, False),
            name='DimShuffleTestTensor')
        return InputTensorType(name='x')

    @pytest.fixture
    def input_layer(self, input_shape, input_var):
        from lasagne.layers.input import InputLayer
        return InputLayer(input_shape, input_var)

    @pytest.fixture
    def input_layer_with_None(self, input_shape_with_None, input_var):
        from lasagne.layers.input import InputLayer
        return InputLayer(input_shape_with_None, input_var)

    @pytest.fixture
    def input_data(self, input_shape):
        return numpy.ones(input_shape)

    def test_rearrange(self, input_data, input_var, input_layer):
        from lasagne.layers.shape import DimshuffleLayer
        ds = DimshuffleLayer(input_layer, [4, 3, 2, 1, 0])
        assert ds.output_shape == (7, 5, 1, 3, 2)
        assert ds.get_output_for(input_var).eval(
            {input_var: input_data}).shape == (7, 5, 1, 3, 2)

    def test_broadcast(self, input_data, input_var, input_layer):
        from lasagne.layers.shape import DimshuffleLayer
        ds = DimshuffleLayer(input_layer, [0, 1, 2, 3, 4, 'x'])
        assert ds.output_shape == (2, 3, 1, 5, 7, 1)
        assert ds.get_output_for(input_var).eval(
            {input_var: input_data}).shape == (2, 3, 1, 5, 7, 1)

    def test_collapse(self, input_data, input_var, input_layer):
        from lasagne.layers.shape import DimshuffleLayer
        ds_ok = DimshuffleLayer(input_layer, [0, 1, 3, 4])
        assert ds_ok.output_shape == (2, 3, 5, 7)
        assert ds_ok.get_output_for(input_var).eval(
            {input_var: input_data}).shape == (2, 3, 5, 7)
        with pytest.raises(ValueError):
            DimshuffleLayer(input_layer, [0, 1, 2, 4])

    def test_collapse_None(self, input_data, input_var, input_layer_with_None):
        from lasagne.layers.shape import DimshuffleLayer
        ds_ok = DimshuffleLayer(input_layer_with_None, [0, 1, 3, 4])
        assert ds_ok.output_shape == (2, 3, 5, 7)
        assert ds_ok.get_output_for(input_var).eval(
            {input_var: input_data}).shape == (2, 3, 5, 7)
        with pytest.raises(ValueError):
            DimshuffleLayer(input_layer_with_None, [0, 1, 2, 4])

    def test_invalid_pattern(self, input_data, input_var, input_layer):
        from lasagne.layers.shape import DimshuffleLayer
        with pytest.raises(ValueError):
            DimshuffleLayer(input_layer, ['q'])
        with pytest.raises(ValueError):
            DimshuffleLayer(input_layer, [0, 0, 1, 3, 4])
        with pytest.raises(ValueError):
            # There is no dimension 42
            DimshuffleLayer(input_layer, [0, 1, 2, 4, 42])
