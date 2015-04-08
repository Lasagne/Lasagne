from mock import Mock
import numpy as np
import pytest
import theano


def max_pool_1d(data, ds, stride=None, pad=0):
    stride = ds if stride is None else stride

    pads = [(0, 0), ] * len(data.shape)
    pads[-1] = (pad, pad)
    data = np.pad(data, pads, mode='constant', constant_values=(-np.inf,))

    data_shifted = np.zeros((ds,) + data.shape)
    data_shifted = data_shifted[..., :data.shape[-1] - ds + 1]
    for i in range(ds):
        data_shifted[i] = data[..., i:i + data.shape[-1] - ds + 1]
    data_pooled = data_shifted.max(axis=0)

    if stride:
        data_pooled = data_pooled[..., ::stride]

    return data_pooled


def max_pool_2d(data, ds, stride, pad):
    data_pooled = max_pool_1d(data, ds[1], stride[1], pad[1])

    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = max_pool_1d(data_pooled, ds[0], stride[0], pad[0])
    data_pooled = np.swapaxes(data_pooled, -1, -2)

    return data_pooled


def pool_test_sets():
    for ds in [2, 3]:
        for stride in [1, 2, 3, 4]:
            for pad in range(ds):
                yield (ds, stride, pad)


class TestMaxPool1DLayer:

    def input_layer(self, output_shape):
        return Mock(get_output_shape=lambda: output_shape)

    def layer_ignoreborder(self, input_layer, ds, stride=None, pad=0):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(
            input_layer,
            ds=ds,
            stride=stride,
            pad=pad,
            ignore_border=True,
        )

    @pytest.mark.parametrize(
        "ds, stride, pad", list(pool_test_sets()))
    def test_get_output_for(self, ds, stride, pad):
        input = np.random.randn(8, 16, 23)
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        layer_output = self.layer_ignoreborder(
            input_layer, ds, stride, pad).get_output_for(input_theano)
        layer_result = layer_output.eval()
        numpy_result = max_pool_1d(input, ds, stride, pad)
        assert np.all(numpy_result.shape == layer_result.shape)
        assert np.allclose(numpy_result, layer_result)

    @pytest.mark.parametrize(
        "input_shape", [(32, 64, 128), (None, 64, 128), (32, None, 128)])
    def test_get_output_shape_for(self, input_shape):
        input_layer = self.input_layer(input_shape)
        layer = self.layer_ignoreborder(input_layer, ds=2)
        assert layer.get_output_shape_for((None, 64, 128)) == (None, 64, 64)
        assert layer.get_output_shape_for((32, 64, 128)) == (32, 64, 64)


class TestMaxPool2DLayer:

    def input_layer(self, output_shape):
        return Mock(get_output_shape=lambda: output_shape)

    def layer_ignoreborder(self, input_layer, ds, stride=None, pad=(0, 0)):
        from lasagne.layers.pool import MaxPool2DLayer
        return MaxPool2DLayer(
            input_layer,
            ds=ds,
            stride=stride,
            pad=pad,
            ignore_border=True,
        )

    @pytest.mark.parametrize(
        "ds, stride, pad", list(pool_test_sets()))
    def test_get_output_for(self, ds, stride, pad):
        input = np.random.randn(8, 16, 17, 13)
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        result = self.layer_ignoreborder(
            input_layer,
            (ds, ds),
            (stride, stride),
            (pad, pad),
        ).get_output_for(input_theano)

        result_eval = result.eval()

        numpy_result = max_pool_2d(
            input, (ds, ds), (stride, stride), (pad, pad))

        assert np.all(numpy_result.shape == result_eval.shape)
        assert np.allclose(result_eval, numpy_result)

    @pytest.mark.parametrize(
        "input_shape",
        [(32, 64, 24, 24), (None, 64, 24, 24), (32, None, 24, 24)],
    )
    def test_get_output_shape_for(self, input_shape):
        input_layer = self.input_layer(input_shape)
        layer = self.layer_ignoreborder(input_layer, ds=(2, 2))
        assert layer.get_output_shape_for(
            (None, 64, 24, 24)) == (None, 64, 12, 12)
        assert layer.get_output_shape_for(
            (32, 64, 24, 24)) == (32, 64, 12, 12)
