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


def upscale_1d_shape(shape, scale_factor):
    return (shape[0], shape[1],
            shape[2] * scale_factor[0])


def upscale_1d(data, scale_factor):
    upscaled = np.zeros(upscale_1d_shape(data.shape, scale_factor))
    for i in range(scale_factor[0]):
        upscaled[:, :, i::scale_factor[0]] = data
    return upscaled


def upscale_1d_dilate(data, scale_factor):
    upscaled = np.zeros(upscale_1d_shape(data.shape, scale_factor))
    upscaled[:, :, ::scale_factor[0]] = data
    return upscaled


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


def max_pool_3d_ignoreborder(data, pool_size, stride, pad):
    # Pool last dim
    data_pooled = max_pool_1d_ignoreborder(
        data, pool_size[2], stride[2], pad[2])
    # Swap second to last to back and pool it
    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = max_pool_1d_ignoreborder(
        data_pooled, pool_size[1], stride[1], pad[1])

    # Swap third to last and pool
    data_pooled = np.swapaxes(data_pooled, -1, -3)
    data_pooled = max_pool_1d_ignoreborder(
        data_pooled, pool_size[0], stride[0], pad[0])

    # Bring back in order
    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = np.swapaxes(data_pooled, -2, -3)

    return data_pooled


def upscale_2d_shape(shape, scale_factor):
    return (shape[0], shape[1],
            shape[2] * scale_factor[0], shape[3] * scale_factor[1])


def upscale_2d(data, scale_factor):
    upscaled = np.zeros(upscale_2d_shape(data.shape, scale_factor))
    for j in range(scale_factor[0]):
        for i in range(scale_factor[1]):
            upscaled[:, :, j::scale_factor[0], i::scale_factor[1]] = data
    return upscaled


def upscale_2d_dilate(data, scale_factor):
    upscaled = np.zeros(upscale_2d_shape(data.shape, scale_factor))
    upscaled[:, :, ::scale_factor[0], ::scale_factor[1]] = data
    return upscaled


def upscale_3d_shape(shape, scale_factor):
    return (shape[0], shape[1],
            shape[2] * scale_factor[0], shape[3] * scale_factor[1],
            shape[4] * scale_factor[2])


def upscale_3d(data, scale_factor):
    upscaled = np.zeros(upscale_3d_shape(data.shape, scale_factor))
    for j in range(scale_factor[0]):
        for i in range(scale_factor[1]):
            for k in range(scale_factor[2]):
                upscaled[:, :, j::scale_factor[0], i::scale_factor[1],
                         k::scale_factor[2]] = data
    return upscaled


def upscale_3d_dilate(data, scale_factor):
    upscaled = np.zeros(upscale_3d_shape(data.shape, scale_factor))
    upscaled[:, :, ::scale_factor[0],
             ::scale_factor[1], ::scale_factor[2]] = data
    return upscaled


def spatial_pool(data, pool_dims):

    def ceildiv(a, b):
        return (a + b - 1) // b

    def floordiv(a, b):
        return a // b

    input_size = data.shape[2:]
    pooled_data_list = []
    for pool_dim in pool_dims:
        pool_size = tuple(ceildiv(i, pool_dim) for i in input_size)
        stride_size = tuple(floordiv(i, pool_dim) for i in input_size)

        pooled_part = max_pool_2d_ignoreborder(
                data, pool_size, stride_size, (0, 0))
        pooled_part = pooled_part.reshape(
                data.shape[0], data.shape[1], pool_dim ** 2)
        pooled_data_list.append(pooled_part)

    return np.concatenate(pooled_data_list, axis=2)


def np_pool_fixed_output_size(feature_maps, output_size, pool_op):
    m, c, h, w = feature_maps.shape
    result = np.zeros((m, c, output_size, output_size),
                      dtype=feature_maps.dtype)

    n = float(output_size)
    for i in range(output_size):
        for j in range(output_size):
            start_h = int(np.floor((j)/n*h))
            end_h = int(np.ceil((j+1)/n*h))
            start_w = int(np.floor((i)/n*w))
            end_w = int(np.ceil((i+1)/n*w))

            region = feature_maps[:, :, start_h:end_h, start_w:end_w]
            result[:, :, j, i] = pool_op(region, axis=(2, 3))
    return result


def np_spatial_pool_kaiming(feature_maps, pool_sizes, mode):
    m, c = feature_maps.shape[0:2]

    if mode == 'max':
        op = np.max
    else:
        op = np.mean

    maps = []
    for p in pool_sizes:
        pool_result = np_pool_fixed_output_size(feature_maps, p, op)
        maps.append(pool_result.reshape((m, c, -1)))
    return np.concatenate(maps, axis=2)


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

        assert np.all(numpy_result.shape == layer.output_shape)
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

    def test_fail_on_mismatching_dimensionality(self):
        from lasagne.layers.pool import MaxPool1DLayer
        with pytest.raises(ValueError) as exc:
            MaxPool1DLayer((10, 20), 3, 2)
        assert "Expected 3 input dimensions" in exc.value.args[0]
        with pytest.raises(ValueError) as exc:
            MaxPool1DLayer((10, 20, 30, 40), 3, 2)
        assert "Expected 3 input dimensions" in exc.value.args[0]


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
        return Mock(output_shape=output_shape)

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

    def test_fail_on_mismatching_dimensionality(self):
        from lasagne.layers.pool import MaxPool2DLayer
        with pytest.raises(ValueError) as exc:
            MaxPool2DLayer((10, 20, 30), 3, 2)
        assert "Expected 4 input dimensions" in exc.value.args[0]
        with pytest.raises(ValueError) as exc:
            MaxPool2DLayer((10, 20, 30, 40, 50), 3, 2)
        assert "Expected 4 input dimensions" in exc.value.args[0]


class TestMaxPool2DCCLayer:
    def pool_test_sets():
        for pool_size in [2, 3]:
            for stride in range(1, pool_size+1):
                yield (pool_size, stride)

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

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

        with pytest.raises(NotImplementedError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2, pad=2)
        assert "MaxPool2DCCLayer does not support padding" in exc.value.args[0]

        with pytest.raises(NotImplementedError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=(2, 3))
        assert ("MaxPool2DCCLayer only supports square pooling regions" in
                exc.value.args[0])

        with pytest.raises(NotImplementedError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2, stride=(1, 2))
        assert (("MaxPool2DCCLayer only supports using the same stride in "
                 "both directions") in exc.value.args[0])

        with pytest.raises(NotImplementedError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2, stride=3)
        assert ("MaxPool2DCCLayer only supports stride <= pool_size" in
                exc.value.args[0])

        with pytest.raises(NotImplementedError) as exc:
            layer = MaxPool2DCCLayer(input_layer, pool_size=2,
                                     ignore_border=True)
        assert ("MaxPool2DCCLayer does not support ignore_border=True" in
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
        return Mock(output_shape=output_shape)

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

    def test_not_implemented(self):
        try:
            from lasagne.layers.dnn import MaxPool2DDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")
        with pytest.raises(NotImplementedError) as exc:
            layer = MaxPool2DDNNLayer((1, 2, 3, 4), pool_size=2,
                                      ignore_border=False)
        assert ("Pool2DDNNLayer does not support ignore_border=False" in
                exc.value.args[0])

    def test_fail_on_mismatching_dimensionality(self):
        try:
            from lasagne.layers.dnn import MaxPool2DDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")
        with pytest.raises(ValueError) as exc:
            MaxPool2DDNNLayer((10, 20, 30), 3, 2)
        assert "Expected 4 input dimensions" in exc.value.args[0]
        with pytest.raises(ValueError) as exc:
            MaxPool2DDNNLayer((10, 20, 30, 40, 50), 3, 2)
        assert "Expected 4 input dimensions" in exc.value.args[0]


class TestMaxPool3DNNLayer:
    def pool_test_sets_ignoreborder():
        for pool_size in [2, 3]:
            for stride in [1, 2, 3, 4]:
                for pad in range(pool_size):
                    yield (pool_size, stride, pad)

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

    def layer(self, input_layer, pool_size, stride, pad):
        try:
            from lasagne.layers.dnn import MaxPool3DDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")

        return MaxPool3DDNNLayer(
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
            input = floatX(np.random.randn(5, 8, 16, 17, 13))
            input_layer = self.input_layer(input.shape)
            input_theano = theano.shared(input)

            result = self.layer(
                input_layer,
                pool_size,
                stride,
                pad,
            ).get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = max_pool_3d_ignoreborder(
                input, [pool_size]*3, [stride]*3, [pad]*3)

            assert np.all(numpy_result.shape == result_eval.shape)
            assert np.allclose(result_eval, numpy_result)
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 32, 64, 24, 24), (32, 32, 32, 12, 12)),
         ((None, 32, 48, 24, 24), (None, 32, 24, 12, 12)),
         ((32, None, 32, 24, 24), (32, None, 16, 12, 12)),
         ((32, 64, None, 24, 24), (32, 64, None, 12, 12)),
         ((32, 64, 32, None, 24), (32, 64, 16, None, 12)),
         ((32, 64, 32, 24, None), (32, 64, 16, 12, None)),
         ((32, 64, 12, None, None), (32, 64, 6, None, None)),
         ((32, 64, None, None, None), (32, 64, None, None, None))],
    )
    def test_get_output_shape_for(self, input_shape, output_shape):
        try:
            input_layer = self.input_layer(input_shape)
            layer = self.layer(input_layer,
                               pool_size=(2, 2, 2), stride=None, pad=(0, 0, 0))
            assert layer.get_output_shape_for(
                input_shape) == output_shape
        except NotImplementedError:
            raise
        #    pytest.skip()

    def test_not_implemented(self):
        try:
            from lasagne.layers.dnn import MaxPool3DDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")
        with pytest.raises(NotImplementedError) as exc:
            layer = MaxPool3DDNNLayer((1, 2, 3, 4, 5), pool_size=2,
                                      ignore_border=False)
        assert ("Pool3DDNNLayer does not support ignore_border=False" in
                exc.value.args[0])

    def test_fail_on_mismatching_dimensionality(self):
        try:
            from lasagne.layers.dnn import MaxPool3DDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")
        with pytest.raises(ValueError) as exc:
            MaxPool3DDNNLayer((10, 20, 30, 40), 3, 2)
        assert "Expected 5 input dimensions" in exc.value.args[0]
        with pytest.raises(ValueError) as exc:
            MaxPool3DDNNLayer((10, 20, 30, 40, 50, 60), 3, 2)
        assert "Expected 5 input dimensions" in exc.value.args[0]


class TestUpscale1DLayer:
    def scale_factor_test_sets():
        for scale_factor in [2, 3]:
            yield scale_factor

    def mode_test_sets():
        for mode in ['repeat', 'dilate']:
            yield mode

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

    def layer(self, input_layer, scale_factor, mode):
        from lasagne.layers.pool import Upscale1DLayer
        return Upscale1DLayer(
            input_layer,
            scale_factor=scale_factor,
            mode=mode,
        )

    def test_invalid_scale_factor(self):
        from lasagne.layers.pool import Upscale1DLayer
        inlayer = self.input_layer((128, 3, 32))
        with pytest.raises(ValueError):
            Upscale1DLayer(inlayer, scale_factor=0)
        with pytest.raises(ValueError):
            Upscale1DLayer(inlayer, scale_factor=-1)
        with pytest.raises(ValueError):
            Upscale1DLayer(inlayer, scale_factor=(0))

    def test_invalid_mode(self):
        from lasagne.layers.pool import Upscale1DLayer
        inlayer = self.input_layer((128, 3, 32))
        with pytest.raises(ValueError):
            Upscale1DLayer(inlayer, scale_factor=1, mode='')
        with pytest.raises(ValueError):
            Upscale1DLayer(inlayer, scale_factor=1, mode='other')
        with pytest.raises(ValueError):
            Upscale1DLayer(inlayer, scale_factor=1, mode=0)

    @pytest.mark.parametrize(
        "scale_factor", list(scale_factor_test_sets()))
    @pytest.mark.parametrize(
        "mode", list(mode_test_sets()))
    def test_get_output_for(self, scale_factor, mode):
        input = floatX(np.random.randn(8, 16, 17))
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        result = self.layer(
            input_layer,
            (scale_factor),
            mode,
        ).get_output_for(input_theano)

        result_eval = result.eval()
        if mode in {'repeat', None}:
            numpy_result = upscale_1d(input, (scale_factor, scale_factor))
        elif mode == 'dilate':
            numpy_result = upscale_1d_dilate(input, (scale_factor,
                                                     scale_factor))

        assert np.all(numpy_result.shape == result_eval.shape)
        assert np.allclose(result_eval, numpy_result)

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24), (32, 64, 48)),
         ((None, 64, 24), (None, 64, 48)),
         ((32, None, 24), (32, None, 48)),
         ((32, 64, None), (32, 64, None))],
    )
    @pytest.mark.parametrize(
        "mode", list(mode_test_sets()))
    def test_get_output_shape_for(self, input_shape, output_shape, mode):
        input_layer = self.input_layer(input_shape)
        layer = self.layer(input_layer,
                           scale_factor=(2),
                           mode=mode)
        assert layer.get_output_shape_for(
            input_shape) == output_shape


class TestUpscale2DLayer:
    def scale_factor_test_sets():
        for scale_factor in [2, 3]:
                yield scale_factor

    def mode_test_sets():
        for mode in ['repeat', 'dilate']:
            yield mode

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

    def layer(self, input_layer, scale_factor, mode):
        from lasagne.layers.pool import Upscale2DLayer
        return Upscale2DLayer(
            input_layer,
            scale_factor=scale_factor,
            mode=mode,
        )

    def test_invalid_scale_factor(self):
        from lasagne.layers.pool import Upscale2DLayer
        inlayer = self.input_layer((128, 3, 32, 32))
        with pytest.raises(ValueError):
            Upscale2DLayer(inlayer, scale_factor=0)
        with pytest.raises(ValueError):
            Upscale2DLayer(inlayer, scale_factor=-1)
        with pytest.raises(ValueError):
            Upscale2DLayer(inlayer, scale_factor=(0, 2))
        with pytest.raises(ValueError):
            Upscale2DLayer(inlayer, scale_factor=(2, 0))

    def test_invalid_mode(self):
        from lasagne.layers.pool import Upscale2DLayer
        inlayer = self.input_layer((128, 3, 32, 32))
        with pytest.raises(ValueError):
            Upscale2DLayer(inlayer, scale_factor=1, mode='')
        with pytest.raises(ValueError):
            Upscale2DLayer(inlayer, scale_factor=1, mode='other')
        with pytest.raises(ValueError):
            Upscale2DLayer(inlayer, scale_factor=1, mode=0)

    @pytest.mark.parametrize(
        "scale_factor", list(scale_factor_test_sets()))
    @pytest.mark.parametrize(
        "mode", list(mode_test_sets()))
    def test_get_output_for(self, scale_factor, mode):
        input = floatX(np.random.randn(8, 16, 17, 13))
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        result = self.layer(
            input_layer,
            (scale_factor, scale_factor),
            mode,
        ).get_output_for(input_theano)

        result_eval = result.eval()
        if mode in {'repeat', None}:
            numpy_result = upscale_2d(input, (scale_factor, scale_factor))
        elif mode == 'dilate':
            numpy_result = upscale_2d_dilate(input, (scale_factor,
                                                     scale_factor))

        assert np.all(numpy_result.shape == result_eval.shape)
        assert np.allclose(result_eval, numpy_result)

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24, 24), (32, 64, 48, 48)),
         ((None, 64, 24, 24), (None, 64, 48, 48)),
         ((32, None, 24, 24), (32, None, 48, 48)),
         ((32, 64, None, 24), (32, 64, None, 48)),
         ((32, 64, 24, None), (32, 64, 48, None)),
         ((32, 64, None, None), (32, 64, None, None))],
    )
    @pytest.mark.parametrize(
        "mode", list(mode_test_sets()))
    def test_get_output_shape_for(self, input_shape, output_shape, mode):
        input_layer = self.input_layer(input_shape)
        layer = self.layer(input_layer,
                           scale_factor=(2, 2),
                           mode=mode)
        assert layer.get_output_shape_for(
            input_shape) == output_shape


class TestUpscale3DLayer:
    def scale_factor_test_sets():
        for scale_factor in [2, 3]:
                yield scale_factor

    def mode_test_sets():
        for mode in ['repeat', 'dilate']:
            yield mode

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

    def layer(self, input_layer, scale_factor, mode):
        from lasagne.layers.pool import Upscale3DLayer
        return Upscale3DLayer(
            input_layer,
            scale_factor=scale_factor,
            mode=mode,
        )

    def test_invalid_scale_factor(self):
        from lasagne.layers.pool import Upscale3DLayer
        inlayer = self.input_layer((128, 3, 32, 32, 32))
        with pytest.raises(ValueError):
            Upscale3DLayer(inlayer, scale_factor=0)
        with pytest.raises(ValueError):
            Upscale3DLayer(inlayer, scale_factor=-1)
        with pytest.raises(ValueError):
            Upscale3DLayer(inlayer, scale_factor=(0, 2, 0))
        with pytest.raises(ValueError):
            Upscale3DLayer(inlayer, scale_factor=(2, 0, -1))

    def test_invalid_mode(self):
        from lasagne.layers.pool import Upscale3DLayer
        inlayer = self.input_layer((128, 3, 32, 32, 32))
        with pytest.raises(ValueError):
            Upscale3DLayer(inlayer, scale_factor=1, mode='')
        with pytest.raises(ValueError):
            Upscale3DLayer(inlayer, scale_factor=1, mode='other')
        with pytest.raises(ValueError):
            Upscale3DLayer(inlayer, scale_factor=1, mode=0)

    @pytest.mark.parametrize(
        "scale_factor", list(scale_factor_test_sets()))
    @pytest.mark.parametrize(
        "mode", list(mode_test_sets()))
    def test_get_output_for(self, scale_factor, mode):
        input = floatX(np.random.randn(8, 16, 17, 13, 15))
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        result = self.layer(
            input_layer,
            (scale_factor, scale_factor, scale_factor),
            mode,
        ).get_output_for(input_theano)

        result_eval = result.eval()
        if mode in {'repeat', None}:
            numpy_result = upscale_3d(input, (scale_factor, scale_factor,
                                              scale_factor))
        elif mode == 'dilate':
            numpy_result = upscale_3d_dilate(input, (scale_factor,
                                                     scale_factor,
                                                     scale_factor))

        assert np.all(numpy_result.shape == result_eval.shape)
        assert np.allclose(result_eval, numpy_result)

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24, 24, 24), (32, 64, 48, 48, 48)),
         ((None, 64, 24, 24, 24), (None, 64, 48, 48, 48)),
         ((32, None, 24, 24, 24), (32, None, 48, 48, 48)),
         ((32, 64, None, 24, 24), (32, 64, None, 48, 48)),
         ((32, 64, 24, None, 24), (32, 64, 48, None, 48)),
         ((32, 64, None, None, 24), (32, 64, None, None, 48)),
         ((32, 64, None, None, None), (32, 64, None, None, None)),
         ((32, 64, 24, 24, None), (32, 64, 48, 48, None))]
    )
    @pytest.mark.parametrize(
        "mode", list(mode_test_sets()))
    def test_get_output_shape_for(self, input_shape, output_shape, mode):
        input_layer = self.input_layer(input_shape)
        layer = self.layer(input_layer,
                           scale_factor=(2, 2, 2),
                           mode=mode)
        assert layer.get_output_shape_for(
            input_shape) == output_shape


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


class TestSpatialPyramidPoolingDNNLayer:
    def pool_dims_test_sets():
        for pyramid_level in [2, 3, 4]:
            pool_dims = list(range(1, pyramid_level))
            yield pool_dims

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

    def layer(self, input_layer, pool_dims):
        try:
            from lasagne.layers.dnn import SpatialPyramidPoolingDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")

        return SpatialPyramidPoolingDNNLayer(input_layer, pool_dims=pool_dims)

    @pytest.mark.parametrize(
        "pool_dims", list(pool_dims_test_sets()))
    @pytest.mark.parametrize(
        "fixed", [True, False])
    def test_get_output_for(self, pool_dims, fixed):
        try:
            input = floatX(np.random.randn(8, 16, 17, 13))
            if fixed:
                input_layer = self.input_layer(input.shape)
            else:
                input_layer = self.input_layer((None, None, None, None))
            input_theano = theano.shared(input)
            layer = self.layer(input_layer, pool_dims)

            result = layer.get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = spatial_pool(input, pool_dims)

            assert result_eval.shape == numpy_result.shape
            assert np.allclose(result_eval, numpy_result)
            assert result_eval.shape[2] == layer.output_shape[2]
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24, 24), (32, 64, 21)),
         ((None, 64, 23, 25), (None, 64, 21)),
         ((32, None, 22, 26), (32, None, 21)),
         ((None, None, None, None), (None, None, 21))],
    )
    def test_get_output_shape_for(self, input_shape, output_shape):
        try:
            input_layer = self.input_layer(input_shape)
            layer = self.layer(input_layer, pool_dims=[1, 2, 4])
            assert layer.get_output_shape_for(input_shape) == output_shape
        except NotImplementedError:
            raise

    def test_fail_on_mismatching_dimensionality(self):
        try:
            from lasagne.layers.dnn import SpatialPyramidPoolingDNNLayer
        except ImportError:
            pytest.skip("cuDNN not available")
        with pytest.raises(ValueError) as exc:
            SpatialPyramidPoolingDNNLayer((10, 20, 30))
        assert "Expected 4 input dimensions" in exc.value.args[0]
        with pytest.raises(ValueError) as exc:
            SpatialPyramidPoolingDNNLayer((10, 20, 30, 40, 50))
        assert "Expected 4 input dimensions" in exc.value.args[0]


class TestSpatialPyramidPoolingLayer:
    def pool_dims_test_sets():
        for pyramid_level in [2, 3, 4]:
            pool_dims = list(range(1, pyramid_level))
            yield pool_dims

    def input_layer(self, output_shape):
        return Mock(output_shape=output_shape)

    def layer(self, input_layer, pool_dims, mode='max', implementation='fast'):
        from lasagne.layers import SpatialPyramidPoolingLayer

        if implementation != 'kaiming':
            try:
                import theano.tensor as T
                from lasagne.layers.pool import pool_2d
                pool_2d(T.tensor4(),
                        ws=T.ivector(),
                        stride=T.ivector(),
                        ignore_border=True,
                        pad=None)
            except ValueError:
                pytest.skip('Old theano version')

        return SpatialPyramidPoolingLayer(input_layer,
                                          pool_dims=pool_dims,
                                          mode=mode,
                                          implementation=implementation)

    @pytest.mark.parametrize(
        "pool_dims", list(pool_dims_test_sets()))
    @pytest.mark.parametrize(
        "fixed", [True, False])
    def test_get_output_for_fast(self, pool_dims, fixed):
        try:
            input = floatX(np.random.randn(8, 16, 17, 13))
            if fixed:
                input_layer = self.input_layer(input.shape)
            else:
                input_layer = self.input_layer((None, None, None, None))
            input_theano = theano.shared(input)
            layer = self.layer(input_layer, pool_dims)

            result = layer.get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = spatial_pool(input, pool_dims)

            assert result_eval.shape == numpy_result.shape
            assert np.allclose(result_eval, numpy_result)
            assert result_eval.shape[2] == layer.output_shape[2]
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "pool_dims", list(pool_dims_test_sets()))
    @pytest.mark.parametrize(
        "fixed", [True, False])
    @pytest.mark.parametrize(
        "mode", ['max', 'average_exc_pad'])
    def test_get_output_for_kaiming(self, pool_dims, fixed, mode):
        try:
            input = floatX(np.random.randn(8, 16, 17, 13))
            if fixed:
                input_layer = self.input_layer(input.shape)
            else:
                input_layer = self.input_layer((None, None, None, None))
            input_theano = theano.shared(input)
            layer = self.layer(input_layer, pool_dims,
                               mode=mode, implementation='kaiming')

            result = layer.get_output_for(input_theano)

            result_eval = result.eval()
            numpy_result = np_spatial_pool_kaiming(input, pool_dims, mode)

            assert result_eval.shape == numpy_result.shape
            assert np.allclose(result_eval, numpy_result, atol=1e-7)
            assert result_eval.shape[2] == layer.output_shape[2]
        except NotImplementedError:
            pytest.skip()

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [((32, 64, 24, 24), (32, 64, 21)),
         ((None, 64, 23, 25), (None, 64, 21)),
         ((32, None, 22, 26), (32, None, 21)),
         ((None, None, None, None), (None, None, 21))],
    )
    def test_get_output_shape_for(self, input_shape, output_shape):
        try:
            input_layer = self.input_layer(input_shape)
            layer = self.layer(input_layer, pool_dims=[1, 2, 4])
            assert layer.get_output_shape_for(input_shape) == output_shape
        except NotImplementedError:
            raise

    def test_fail_on_mismatching_dimensionality(self):
        from lasagne.layers import SpatialPyramidPoolingLayer

        with pytest.raises(ValueError) as exc:
            SpatialPyramidPoolingLayer((10, 20, 30))
        assert "Expected 4 input dimensions" in exc.value.args[0]
        with pytest.raises(ValueError) as exc:
            SpatialPyramidPoolingLayer((10, 20, 30, 40, 50))
        assert "Expected 4 input dimensions" in exc.value.args[0]

    def test_fail_invalid_mode(self):
        with pytest.raises(ValueError) as exc:
            input = self.input_layer((None, None, None, None))
            layer = self.layer(input, pool_dims=[1],
                               mode='other', implementation='kaiming')
            layer.get_output_for(Mock(shape=(1, 1, 1, 1)))
        assert "Mode must be either 'max', 'average_inc_pad' or " \
               "'average_exc_pad'. Got 'other'" in exc.value.args[0]
