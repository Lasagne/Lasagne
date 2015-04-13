from mock import Mock
import numpy as np
import pytest
import theano
from sets import Set


def max_pool_1d(data, ds, st=None):
    st = ds if st is None else st

    idx = range(data.shape[-1])
    used_idx = Set([])
    idx_sets = []

    i = 0
    while i < data.shape[-1]:
        idx_set = Set(range(i, i + ds))
        idx_set = idx_set.intersection(idx)
        if not idx_set.issubset(used_idx):
            idx_sets.append(list(idx_set))
            used_idx = used_idx.union(idx_set)
        i += st

    data_pooled = np.array(
        [data[..., idx_set].max(axis=-1) for idx_set in idx_sets])
    data_pooled = np.rollaxis(data_pooled, 0, len(data_pooled.shape))

    return data_pooled


def max_pool_1d_ignoreborder(data, ds, st=None, pad=0):
    st = ds if st is None else st

    pads = [(0, 0), ] * len(data.shape)
    pads[-1] = (pad, pad)
    data = np.pad(data, pads, mode='constant', constant_values=(-np.inf,))

    data_shifted = np.zeros((ds,) + data.shape)
    data_shifted = data_shifted[..., :data.shape[-1] - ds + 1]
    for i in range(ds):
        data_shifted[i] = data[..., i:i + data.shape[-1] - ds + 1]
    data_pooled = data_shifted.max(axis=0)

    if st:
        data_pooled = data_pooled[..., ::st]

    return data_pooled


def max_pool_2d(data, ds, st):
    data_pooled = max_pool_1d(data, ds[1], st[1])

    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = max_pool_1d(data_pooled, ds[0], st[0])
    data_pooled = np.swapaxes(data_pooled, -1, -2)

    return data_pooled


def max_pool_2d_ignoreborder(data, ds, st, pad):
    data_pooled = max_pool_1d_ignoreborder(data, ds[1], st[1], pad[1])

    data_pooled = np.swapaxes(data_pooled, -1, -2)
    data_pooled = max_pool_1d_ignoreborder(data_pooled, ds[0], st[0], pad[0])
    data_pooled = np.swapaxes(data_pooled, -1, -2)

    return data_pooled


def pool_test_sets():
    for ds in [2, 3]:
        for st in [1, 2, 3, 4]:
            yield (ds, st)


def pool_test_sets_ignoreborder():
    for ds in [2, 3]:
        for st in [1, 2, 3, 4]:
            for pad in range(ds):
                yield (ds, st, pad)


class TestMaxPool1DLayer:

    def input_layer(self, output_shape):
        return Mock(get_output_shape=lambda: output_shape)

    def layer(self, input_layer, ds, st=None, pad=0):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(
            input_layer,
            ds=ds,
            st=st,
            ignore_border=False,
        )

    def layer_ignoreborder(self, input_layer, ds, st=None, pad=0):
        from lasagne.layers.pool import MaxPool1DLayer
        return MaxPool1DLayer(
            input_layer,
            ds=ds,
            st=st,
            pad=pad,
            ignore_border=True,
        )

    @pytest.mark.parametrize(
        "ds, st", list(pool_test_sets()))
    def test_get_output_for(self, ds, st):
        input = np.random.randn(8, 16, 23)
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        layer_output = self.layer(
            input_layer, ds, st).get_output_for(input_theano)
        layer_result = layer_output.eval()
        numpy_result = max_pool_1d(input, ds, st)
        assert np.all(numpy_result.shape == layer_result.shape)
        assert np.allclose(numpy_result, layer_result)

    @pytest.mark.parametrize(
        "ds, st, pad", list(pool_test_sets_ignoreborder()))
    def test_get_output_for_ignoreborder(self, ds, st, pad):
        input = np.random.randn(8, 16, 23)
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        layer_output = self.layer_ignoreborder(
            input_layer, ds, st, pad).get_output_for(input_theano)
        layer_result = layer_output.eval()
        numpy_result = max_pool_1d_ignoreborder(input, ds, st, pad)
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

    def layer(self, input_layer, ds, st=None):
        from lasagne.layers.pool import MaxPool2DLayer
        return MaxPool2DLayer(
            input_layer,
            ds=ds,
            st=st,
            ignore_border=False,
        )

    def layer_ignoreborder(self, input_layer, ds, st=None, pad=(0, 0)):
        from lasagne.layers.pool import MaxPool2DLayer
        return MaxPool2DLayer(
            input_layer,
            ds=ds,
            st=st,
            pad=pad,
            ignore_border=True,
        )

    @pytest.mark.parametrize(
        "ds, st", list(pool_test_sets()))
    def test_get_output_for(self, ds, st):
        input = np.random.randn(8, 16, 17, 13)
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        result = self.layer(
            input_layer,
            (ds, ds),
            (st, st),
        ).get_output_for(input_theano)

        result_eval = result.eval()

        numpy_result = max_pool_2d(input, (ds, ds), (st, st))

        assert np.all(numpy_result.shape == result_eval.shape)
        assert np.allclose(result_eval, numpy_result)

    @pytest.mark.parametrize(
        "ds, st, pad", list(pool_test_sets_ignoreborder()))
    def test_get_output_for_ignoreborder(self, ds, st, pad):
        input = np.random.randn(8, 16, 17, 13)
        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)
        result = self.layer_ignoreborder(
            input_layer,
            (ds, ds),
            (st, st),
            (pad, pad),
        ).get_output_for(input_theano)

        result_eval = result.eval()

        numpy_result = max_pool_2d_ignoreborder(
            input, (ds, ds), (st, st), (pad, pad))

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
