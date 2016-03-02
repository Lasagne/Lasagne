import numpy as np
import pytest
import theano

from mock import Mock


class TestFlattenLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.shape import FlattenLayer
        return FlattenLayer(Mock(output_shape=(None,)))

    @pytest.fixture
    def layer_outdim3(self):
        from lasagne.layers.shape import FlattenLayer
        return FlattenLayer(Mock(output_shape=(None,)), outdim=3)

    @pytest.fixture
    def layer_outdim1(self):
        from lasagne.layers.shape import FlattenLayer
        return FlattenLayer(Mock(output_shape=(None,)), outdim=1)

    def test_get_output_shape_for(self, layer):
        input_shape = (2, 3, 4, 5)
        assert layer.get_output_shape_for(input_shape) == (2, 3 * 4 * 5)

    def test_get_output_shape_for_contain_none(self, layer):
        input_shape = (2, 3, None, 5)
        assert layer.get_output_shape_for(input_shape) == (2, None)

    def test_get_output_for(self, layer):
        input = np.random.random((2, 3, 4, 5))
        result = layer.get_output_for(theano.shared(input)).eval()
        assert (result == input.reshape((input.shape[0], -1))).all()

    def test_get_output_shape_for_outdim3(self, layer_outdim3):
        input_shape = (2, 3, 4, 5)
        assert layer_outdim3.get_output_shape_for(input_shape) == (2, 3, 4 * 5)

    def test_get_output_for_outdim3(self, layer_outdim3):
        input = np.random.random((2, 3, 4, 5))
        result = layer_outdim3.get_output_for(theano.shared(input)).eval()
        assert (result == input.reshape(
            (input.shape[0], input.shape[1], -1))).all()

    def test_get_output_shape_for_outdim1(self, layer_outdim1):
        input_shape = (2, 3, 4, 5)
        assert layer_outdim1.get_output_shape_for(input_shape) == (
            2 * 3 * 4 * 5, )

    def test_get_output_for_outdim1(self, layer_outdim1):
        input = np.random.random((2, 3, 4, 5))
        result = layer_outdim1.get_output_for(theano.shared(input)).eval()
        assert (result == input.reshape(-1)).all()

    def test_dim0_raises(self):
        from lasagne.layers.shape import FlattenLayer
        with pytest.raises(ValueError):
            FlattenLayer((2, 3, 4), outdim=0)


class TestPadLayer:
    @pytest.fixture
    def layerclass(self):
        from lasagne.layers.shape import PadLayer
        return PadLayer

    @pytest.mark.parametrize(
        "width, input_shape, output_shape",
        [(3, (2, 3, 4, 5), (2, 3, 10, 11)),
         ((2, 3), (2, 3, 4, 5), (2, 3, 8, 11)),
         (((1, 2), (3, 4)), (2, 3, 4, 5), (2, 3, 7, 12)),
         (3, (2, 3, None, 5), (2, 3, None, 11)),
         ((2, 3), (2, 3, 4, None), (2, 3, 8, None)),
         (((1, 2), (3, 4)), (None, 3, None, None), (None, 3, None, None)),
         ])
    def test_get_output_shape_for(self, layerclass,
                                  width, input_shape, output_shape):
        layer = layerclass(Mock(output_shape=(None,)), width=width)
        assert layer.get_output_shape_for(input_shape) == output_shape

    def test_get_output_for(self, layerclass):
        layer = layerclass(Mock(output_shape=(None,)), width=2)
        input = np.zeros((1, 2, 10))
        trimmed = theano.shared(input[:, :, 2:-2])
        result = layer.get_output_for(trimmed).eval()

        assert (result == input).all()


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
                theano.shared(np.ones((16, 3, 5, 7, 10))))

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
        with pytest.raises(ValueError):
            layerclass(inputlayer, (16, 3, theano.tensor.vector(), 7, 10))

    def test_symbolic_shape(self):
        from lasagne.layers import InputLayer, ReshapeLayer, get_output
        x = theano.tensor.tensor3()
        batch_size, seq_len, num_features = x.shape
        l_inp = InputLayer((None, None, None))
        l_rshp2 = ReshapeLayer(l_inp, (batch_size*seq_len, [2]))

        # we cannot infer any of the output shapes because they are symbolic.
        output_shape = l_rshp2.get_output_shape_for(
            (batch_size, seq_len, num_features))
        assert output_shape == (None, None)

        output = get_output(l_rshp2, x)
        out1 = output.eval({x: np.ones((3, 5, 6), dtype='float32')})
        out2 = output.eval({x: np.ones((4, 5, 7), dtype='float32')})

        assert out1.shape == (3*5, 6)
        assert out2.shape == (4*5, 7)


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
    def input_layer_with_None(self, input_shape_with_None, input_var):
        from lasagne.layers.input import InputLayer
        return InputLayer(input_shape_with_None, input_var)

    @pytest.fixture
    def input_data(self, input_shape):
        return np.ones(input_shape)

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


def test_slice_layer():
    from lasagne.layers import SliceLayer, InputLayer, get_output_shape,\
        get_output
    from numpy.testing import assert_array_almost_equal as aeq
    in_shp = (3, 5, 2)
    l_inp = InputLayer(in_shp)
    l_slice_ax0 = SliceLayer(l_inp, axis=0, indices=0)
    l_slice_ax1 = SliceLayer(l_inp, axis=1, indices=slice(3, 5))
    l_slice_ax2 = SliceLayer(l_inp, axis=-1, indices=-1)

    x = np.arange(np.prod(in_shp)).reshape(in_shp).astype('float32')
    x1 = x[0]
    x2 = x[:, 3:5]
    x3 = x[:, :, -1]

    assert get_output_shape(l_slice_ax0) == x1.shape
    assert get_output_shape(l_slice_ax1) == x2.shape
    assert get_output_shape(l_slice_ax2) == x3.shape

    aeq(get_output(l_slice_ax0, x).eval(), x1)
    aeq(get_output(l_slice_ax1, x).eval(), x2)
    aeq(get_output(l_slice_ax2, x).eval(), x3)

    # test slicing None dimension
    in_shp = (2, None, 2)
    l_inp = InputLayer(in_shp)
    l_slice_ax1 = SliceLayer(l_inp, axis=1, indices=slice(3, 5))
    assert get_output_shape(l_slice_ax1) == (2, None, 2)
    aeq(get_output(l_slice_ax1, x).eval(), x2)
