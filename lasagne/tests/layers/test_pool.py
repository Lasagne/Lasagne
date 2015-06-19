from mock import Mock
import numpy as np
import pytest
import theano

from lasagne.utils import floatX


def max_pool_1d(data, pool_size, stride=None):
    stride = pool_size if stride is None else stride

    idx = range(data.shape[-1])
    used_idx = set([])
    idx_sets = []

    i = 0
    while i < data.shape[-1]:
        idx_set = set(range(i, i + pool_size))
        idx_set = idx_set.intersection(idx)
        if not idx_set.issubset(used_idx):
            idx_sets.append(list(idx_set))
            used_idx = used_idx.union(idx_set)
        i += stride

    data_pooled = np.array(
        [data[..., idx_set].max(axis=-1) for idx_set in idx_sets])
    data_pooled = np.rollaxis(data_pooled, 0, len(data_pooled.shape))

    return data_pooled


def max_pool_1d_ignoreborder(data, pool_size, stride=None, pad=0):
    stride = pool_size if stride is None else stride

    pads = [(0, 0), ] * len(data.shape)
    pads[-1] = (pad, pad)
    data = np.pad(data, pads, mode='constant', constant_values=(-np.inf,))

    data_shifted = np.zeros((pool_size,) + data.shape)
    data_shifted = data_shifted[..., :data.shape[-1] - pool_size + 1]
    for i in range(pool_size):
        data_shifted[i] = data[..., i:i + data.shape[-1] - pool_size + 1]
    data_pooled = data_shifted.max(axis=0)

    if stride:
        data_pooled = data_pooled[..., ::stride]

    return data_pooled


def max_pool_2d(data, pool_size, stride):
    data_pooled = max_pool_1d(data, pool_size[1], stride[1])

    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = max_pool_1d(data_pooled, pool_size[0], stride[0])
    data_pooled = np.swapaxes(data_pooled, -1, -2)

    return data_pooled


def max_pool_2d_ignoreborder(data, pool_size, stride, pad):
    data_pooled = max_pool_1d_ignoreborder(
        data, pool_size[1], stride[1], pad[1])

    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = max_pool_1d_ignoreborder(
        data_pooled, pool_size[0], stride[0], pad[0])
    data_pooled = np.swapaxes(data_pooled, -1, -2)

    return data_pooled


class TestFeaturePoolLayer:
    def pool_test_sets():
        for pool_size in [2, 3]:
            for axis in [1, 2]:
                yield (pool_size, axis)

    def input_layer(self, output_shape):
        from lasagne.layers.input import InputLayer
        return InputLayer(output_shape)

    def layer(self, input_layer, pool_size, axis):
        from lasagne.layers.pool import FeaturePoolLayer
        return FeaturePoolLayer(
            input_layer,
            pool_size=pool_size,
            axis=axis,
        )

    def test_init_raises(self):
        input_layer = self.input_layer((2, 3, 4))

        with pytest.raises(ValueError):
            self.layer(input_layer, pool_size=2, axis=1)

    @pytest.mark.parametrize(
        "pool_size, axis", list(pool_test_sets()))
    def test_layer(self, pool_size, axis):
        input = floatX(np.random.randn(3, 6, 12, 23))
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)

        layer = self.layer(input_layer, pool_size, axis)
        layer_result = layer.get_output_for(input_theano).eval()

        numpy_result = np.swapaxes(input, axis, -1)
        numpy_result = max_pool_1d(numpy_result, pool_size)
        numpy_result = np.swapaxes(numpy_result, -1, axis)

        assert np.all(numpy_result.shape == layer.get_output_shape())
        assert np.all(numpy_result.shape == layer_result.shape)
        assert np.allclose(numpy_result, layer_result)


class TestMaxPool1DLayer:
    def pool_test_sets():
        for pool_size in [2, 3]:
            for stride in [1, 2, 3, 4]:
                yield (pool_size, stride)

    def pool_test_sets_ignoreborder():
        for pool_size in [2, 3]:
            for stride in [1, 2, 3, 4]:
                for pad in range(pool_size):
                    yield (pool_size, stride, pad)

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

    def layer(self, input_layer, pool_size, stride=None, pad=0):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(
            input_layer,
            pool_size=pool_size,
            stride=stride,
            ignore_border=False,
        )

    def layer_ignoreborder(self, input_layer, pool_size, stride=None, pad=0):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(
            input_layer,
            pool_size=pool_size,
            stride=stride,
            pad=pad,
            ignore_border=True,
        )

    @pytest.mark.parametrize(
        "pool_size, stride", list(pool_test_sets()))
    def test_get_output_and_shape_for(self, pool_size, stride):
        input = floatX(np.random.randn(8, 16, 23))
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)

        layer = self.layer(input_layer, pool_size, stride)
        layer_output_shape = layer.get_output_shape_for(input.shape)
        layer_output = layer.get_output_for(input_theano)
        layer_result = layer_output.eval()

        numpy_result = max_pool_1d(input, pool_size, stride)

        assert numpy_result.shape == layer_output_shape
        assert np.allclose(numpy_result, layer_result)

    @pytest.mark.parametrize(
        "pool_size, stride, pad", list(pool_test_sets_ignoreborder()))
    def test_get_output_for_ignoreborder(self, pool_size, stride, pad):
        input = floatX(np.random.randn(8, 16, 23))
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        layer_output = self.layer_ignoreborder(
            input_layer, pool_size, stride, pad).get_output_for(input_theano)

        layer_result = layer_output.eval()
        numpy_result = max_pool_1d_ignoreborder(input, pool_size, stride, pad)

        assert np.all(numpy_result.shape == layer_result.shape)
        assert np.allclose(numpy_result, layer_result)

    @pytest.mark.parametrize(
        "input_shape", [(32, 64, 128), (None, 64, 128), (32, None, 128),
                        (32, 64, None)])
    def test_get_output_shape_for(self, input_shape):
        input_layer = self.input_layer(input_shape)
        layer = self.layer_ignoreborder(input_layer, pool_size=2)
        assert layer.get_output_shape_for((None, 64, 128)) == (None, 64, 64)
        assert layer.get_output_shape_for((32, 64, None)) == (32, 64, None)
        assert layer.get_output_shape_for((32, 64, 128)) == (32, 64, 64)


class TestMaxPool2DLayer:
    def pool_test_sets():
        for pool_size in [2, 3]:
            for stride in [1, 2, 3, 4]:
                yield (pool_size, stride)

    def pool_test_sets_ignoreborder():
        for pool_size in [2, 3]:
            for stride in [1, 2, 3, 4]:
                for pad in range(pool_size):
                    yield (pool_size, stride, pad)

    def input_layer(self, output_shape):
        return Mock(get_output_shape=lambda: output_shape,
                    output_shape=output_shape)

    def layer(self, input_layer, pool_size, stride=None,
              pad=(0, 0), ignore_border=False):
        from lasagne.layers.pool import MaxPool2DLayer
        return MaxPool2DLayer(
            input_layer,
            pool_size=pool_size,
            stride=stride,
            pad=pad,
            ignore_border=ignore_border,
        )

    @pytest.mark.parametrize(
        "pool_size, stride", list(pool_test_sets()))
    def test_get_output_for(self, pool_size, stride):
        try:
            input = floatX(np.random.randn(8, 16, 17, 13))
            input_layer = self.input_layer(input.shape)
            input_theano = theano.shared(input)
            result = self.layer(
                input_layer,
                (pool_size, pool_size),
                (stride, stride),
                ignore_border=False,
            ).get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = max_pool_2d(
                input, (pool_size, pool_size), (stride, stride))

            assert np.all(numpy_result.shape == result_eval.shape)
            assert np.allclose(result_eval, numpy_result)
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "pool_size, stride, pad", list(pool_test_sets_ignoreborder()))
    def test_get_output_for_ignoreborder(self, pool_size,
                                         stride, pad):
        try:
            input = floatX(np.random.randn(8, 16, 17, 13))
            input_layer = self.input_layer(input.shape)
            input_theano = theano.shared(input)

            result = self.layer(
                input_layer,
                pool_size,
                stride,
                pad,
                ignore_border=True,
            ).get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = max_pool_2d_ignoreborder(
                input, (pool_size, pool_size), (stride, stride), (pad, pad))

            assert np.all(numpy_result.shape == result_eval.shape)
            assert np.allclose(result_eval, numpy_result)
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24, 24), (32, 64, 12, 12)),
         ((None, 64, 24, 24), (None, 64, 12, 12)),
         ((32, None, 24, 24), (32, None, 12, 12)),
         ((32, 64, None, 24), (32, 64, None, 12)),
         ((32, 64, 24, None), (32, 64, 12, None)),
         ((32, 64, None, None), (32, 64, None, None))],
    )
    def test_get_output_shape_for(self, input_shape, output_shape):
        try:
            input_layer = self.input_layer(input_shape)
            layer = self.layer(input_layer,
                               pool_size=(2, 2), stride=None)
            assert layer.get_output_shape_for(
                input_shape) == output_shape
        except NotImplementedError:
            pytest.skip()


class TestMaxPool2DCCLayer:
    def pool_test_sets():
        for pool_size in [2, 3]:
            for stride in range(1, pool_size+1):
                yield (pool_size, stride)

    def input_layer(self, output_shape):
        return Mock(get_output_shape=lambda: output_shape,
                    output_shape=output_shape)

    def layer(self, input_layer, pool_size, stride):
        try:
            from lasagne.layers.cuda_convnet import MaxPool2DCCLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")
        return MaxPool2DCCLayer(
            input_layer,
            pool_size=pool_size,
            stride=stride,
        )

    @pytest.mark.parametrize(
        "pool_size, stride", list(pool_test_sets()))
    def test_get_output_for(self, pool_size, stride):
        try:
            input = floatX(np.random.randn(8, 16, 16, 16))
            input_layer = self.input_layer(input.shape)
            input_theano = theano.shared(input)
            result = self.layer(
                input_layer,
                (pool_size, pool_size),
                (stride, stride),
            ).get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = max_pool_2d(
                input, (pool_size, pool_size), (stride, stride))

            assert np.all(numpy_result.shape == result_eval.shape)
            assert np.allclose(result_eval, numpy_result)
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24, 24), (32, 64, 12, 12)),
         ((None, 64, 24, 24), (None, 64, 12, 12)),
         ((32, None, 24, 24), (32, None, 12, 12)),
         ((32, 64, None, 24), (32, 64, None, 12)),
         ((32, 64, 24, None), (32, 64, 12, None)),
         ((32, 64, None, None), (32, 64, None, None))],
    )
    def test_get_output_shape_for(self, input_shape, output_shape):
        try:
            input_layer = self.input_layer(input_shape)
            layer = self.layer(input_layer,
                               pool_size=(2, 2), stride=None)
            assert layer.get_output_shape_for(
                input_shape) == output_shape
        except NotImplementedError:
            pytest.skip()

    def test_not_implemented(self):
        try:
            from lasagne.layers.cuda_convnet import MaxPool2DCCLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")

        input_layer = self.input_layer((128, 4, 12, 12))

        with pytest.raises(RuntimeError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2, pad=2)
        assert "MaxPool2DCCLayer does not support padding" in exc.value.args[0]

        with pytest.raises(RuntimeError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=(2, 3))
        assert ("MaxPool2DCCLayer only supports square pooling regions" in
                exc.value.args[0])

        with pytest.raises(RuntimeError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2, stride=(1, 2))
        assert (("MaxPool2DCCLayer only supports using the same stride in "
                 "both directions") in exc.value.args[0])

        with pytest.raises(RuntimeError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2, stride=3)
        assert ("MaxPool2DCCLayer only supports stride <= pool_size" in
                exc.value.args[0])

        with pytest.raises(RuntimeError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2,
                                     ignore_border=True)
        assert ("MaxPool2DCCLayer does not support ignore_border" in
                exc.value.args[0])

    def test_dimshuffle_false(self):
        try:
            from lasagne.layers.cuda_convnet import MaxPool2DCCLayer
        except ImportError:
            pytest.skip("cuda_convnet not available")
        from lasagne.layers.input import InputLayer

        input_layer = InputLayer((4, 12, 12, 16))  # c01b order
        layer = MaxPool2DCCLayer(input_layer, pool_size=2, dimshuffle=False)
        assert layer.output_shape == (4, 6, 6, 16)

        input = floatX(np.random.randn(4, 12, 12, 16))
        output = max_pool_2d(input.transpose(3, 0, 1, 2), (2, 2), (2, 2))
        output = output.transpose(1, 2, 3, 0)
        actual = layer.get_output_for(input).eval()
        assert np.allclose(output, actual)


class TestMaxPool2DNNLayer:
    def pool_test_sets_ignoreborder():
        for pool_size in [2, 3]:
            for stride in [1, 2, 3, 4]:
                for pad in range(pool_size):
                    yield (pool_size, stride, pad)

    def input_layer(self, output_shape):
        return Mock(get_output_shape=lambda: output_shape,
                    output_shape=output_shape)

    def layer(self, input_layer, pool_size, stride, pad):
        try:
            from lasagne.layers.dnn import MaxPool2DDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")

        return MaxPool2DDNNLayer(
            input_layer,
            pool_size=pool_size,
            stride=stride,
            pad=pad,
        )

    @pytest.mark.parametrize(
        "pool_size, stride, pad", list(pool_test_sets_ignoreborder()))
    def test_get_output_for_ignoreborder(self, pool_size,
                                         stride, pad):
        try:
            input = floatX(np.random.randn(8, 16, 17, 13))
            input_layer = self.input_layer(input.shape)
            input_theano = theano.shared(input)

            result = self.layer(
                input_layer,
                pool_size,
                stride,
                pad,
            ).get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = max_pool_2d_ignoreborder(
                input, (pool_size, pool_size), (stride, stride), (pad, pad))

            assert np.all(numpy_result.shape == result_eval.shape)
            assert np.allclose(result_eval, numpy_result)
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24, 24), (32, 64, 12, 12)),
         ((None, 64, 24, 24), (None, 64, 12, 12)),
         ((32, None, 24, 24), (32, None, 12, 12)),
         ((32, 64, None, 24), (32, 64, None, 12)),
         ((32, 64, 24, None), (32, 64, 12, None)),
         ((32, 64, None, None), (32, 64, None, None))],
    )
    def test_get_output_shape_for(self, input_shape, output_shape):
        try:
            input_layer = self.input_layer(input_shape)
            layer = self.layer(input_layer,
                               pool_size=(2, 2), stride=None, pad=(0, 0))
            assert layer.get_output_shape_for(
                input_shape) == output_shape
        except NotImplementedError:
            raise
        #    pytest.skip()


class TestFeatureWTALayer(object):
    @pytest.fixture
    def FeatureWTALayer(self):
        from lasagne.layers.pool import FeatureWTALayer
        return FeatureWTALayer

    @pytest.fixture
    def input_layer(self):
        from lasagne.layers.input import InputLayer
        return InputLayer((2, 4, 8))

    @pytest.fixture
    def layer(self, FeatureWTALayer, input_layer):
        return FeatureWTALayer(input_layer, pool_size=2)

    def test_init_raises(self, FeatureWTALayer, input_layer):
        with pytest.raises(ValueError):
            FeatureWTALayer(input_layer, pool_size=3)

    def test_get_output_for(self, layer):
        input = theano.shared(np.random.uniform(-1, 1, (2, 4, 8)))
        result = layer.get_output_for(input).eval()

        reshaped = input.get_value().reshape((2, 2, 2, 8))
        np_result = reshaped * (reshaped == reshaped.max(2, keepdims=True))
        np_result = np_result.reshape((2, 4, 8))

        assert np.allclose(result, np_result)


class TestGlobalPoolLayer(object):
    @pytest.fixture
    def GlobalPoolLayer(self):
        from lasagne.layers.pool import GlobalPoolLayer
        return GlobalPoolLayer

    @pytest.fixture
    def layer(self, GlobalPoolLayer):
        return GlobalPoolLayer(Mock(output_shape=(None,)))

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for((2, 3, 4, 5)) == (2, 3)

    def test_get_output_for(self, layer):
        input = theano.shared(np.random.uniform(-1, 1, (2, 3, 4, 5)))
        result = layer.get_output_for(input).eval()

        np_result = input.get_value().reshape((2, 3, -1)).mean(-1)

        assert np.allclose(result, np_result)
