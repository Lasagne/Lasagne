import numpy as np
import pytest
import theano

from lasagne.utils import floatX


def locally_connected2d(input, W, flip_filters=True):
    """
    2D convolution with unshared weights, no stride, 'same' padding,
    no dilation and no bias
    """
    num_batch, input_channels, input_rows, input_cols = input.shape
    assert W.shape[1] == input_channels
    num_filters, input_channels, \
        filter_rows, filter_cols, output_rows, output_cols = W.shape
    assert filter_rows % 2 == 1
    assert filter_cols % 2 == 1
    output = np.zeros((num_batch, num_filters, output_rows, output_cols))
    for b in range(num_batch):
        for f in range(num_filters):
            for c in range(input_channels):
                for i_out in range(output_rows):
                    for j_out in range(output_cols):
                        for i_filter in range(filter_rows):
                            i_in = i_out + i_filter - (filter_rows // 2)
                            if not (0 <= i_in < input_rows):
                                continue
                            for j_filter in range(filter_cols):
                                j_in = j_out + j_filter - (filter_cols // 2)
                                if not (0 <= j_in < input_cols):
                                    continue
                                if flip_filters:
                                    inc = (input[b, c, i_in, j_in] *
                                           W[f, c, -i_filter-1, -j_filter-1,
                                             i_out, j_out])
                                else:
                                    inc = (input[b, c, i_in, j_in] *
                                           W[f, c, i_filter, j_filter,
                                             i_out, j_out])
                                output[b, f, i_out, j_out] += inc
    return output


def channelwise_locally_connected2d(input, W, flip_filters=True):
    """
    channelwise 2D convolution with unshared weights, no stride,
    'same' padding, no dilation and no bias
    """
    num_batch, input_channels, input_rows, input_cols = input.shape
    num_filters, filter_rows, filter_cols, output_rows, output_cols = W.shape
    assert input_channels == num_filters
    assert filter_rows % 2 == 1
    assert filter_cols % 2 == 1
    output = np.zeros((num_batch, num_filters, output_rows, output_cols))
    for b in range(num_batch):
        for f in range(num_filters):
            for i_out in range(output_rows):
                for j_out in range(output_cols):
                    for i_filter in range(filter_rows):
                        i_in = i_out + i_filter - (filter_rows // 2)
                        if not (0 <= i_in < input_rows):
                            continue
                        for j_filter in range(filter_cols):
                            j_in = j_out + j_filter - (filter_cols // 2)
                            if not (0 <= j_in < input_cols):
                                continue
                            if flip_filters:
                                inc = (input[b, f, i_in, j_in] *
                                       W[f, -i_filter-1, -j_filter-1,
                                         i_out, j_out])
                            else:
                                inc = (input[b, f, i_in, j_in] *
                                       W[f, i_filter, j_filter,
                                         i_out, j_out])
                            output[b, f, i_out, j_out] += inc
    return output


def locally_connected2d_test_sets():
    def _convert(input, W, output, kwargs):
        return [floatX(input), floatX(W), output, kwargs]

    for batch_size in (2, 3):
        for input_shape in ((batch_size, 2, 5, 5), (batch_size, 4, 8, 8)):
            for num_filters in (2, 4):
                for filter_size in ((3, 3), (3, 5)):
                    for flip_filters in (True, False):
                        for channelwise in (True, False):
                            if channelwise and num_filters != input_shape[1]:
                                continue
                            input = np.random.random(input_shape)
                            if channelwise:
                                W = np.random.random(
                                    (num_filters,) + filter_size +
                                    input_shape[2:])
                                output = channelwise_locally_connected2d(
                                    input, W, flip_filters=flip_filters)
                            else:
                                W = np.random.random(
                                    (num_filters, input_shape[1]) +
                                    filter_size + input_shape[2:])
                                output = locally_connected2d(
                                    input, W, flip_filters=flip_filters)
                            yield _convert(input, W, output,
                                           {'num_filters': num_filters,
                                            'filter_size': filter_size,
                                            'flip_filters': flip_filters,
                                            'channelwise': channelwise})


@pytest.fixture
def DummyInputLayer():
    def factory(shape):
        from lasagne.layers.input import InputLayer
        return InputLayer(shape)
    return factory


class TestLocallyConnected2DLayer:
    @pytest.mark.parametrize(
        "input, W, output, kwargs", list(locally_connected2d_test_sets()))
    def test_defaults(self, DummyInputLayer, input, W, output, kwargs):
        from lasagne.layers import LocallyConnected2DLayer
        b, c, h, w = input.shape
        input_layer = DummyInputLayer((b, c, h, w))
        layer = LocallyConnected2DLayer(
                input_layer,
                W=W,
                **kwargs)
        actual = layer.get_output_for(theano.shared(input)).eval()
        assert actual.shape == output.shape
        assert actual.shape == layer.output_shape
        assert np.allclose(actual, output)

    def test_unsupported_settings(self, DummyInputLayer):
        from lasagne.layers import LocallyConnected2DLayer
        input_layer = DummyInputLayer((10, 2, 4, 4))
        for pad in 'valid', 'full', 1:
            with pytest.raises(NotImplementedError) as exc:
                LocallyConnected2DLayer(input_layer, 2, 3, pad=pad)
            assert "requires pad='same'" in exc.value.args[0]
        with pytest.raises(NotImplementedError) as exc:
            LocallyConnected2DLayer(input_layer, 2, 3, stride=2)
        assert "requires stride=1 / (1, 1)" in exc.value.args[0]

    def test_invalid_settings(self, DummyInputLayer):
        from lasagne.layers import LocallyConnected2DLayer
        input_layer = DummyInputLayer((10, 2, 4, 4))
        with pytest.raises(ValueError) as exc:
            LocallyConnected2DLayer(input_layer, 4, 3, channelwise=True)
        assert "num_filters and the number of input channels should match" \
               in exc.value.args[0]
        input_layer = DummyInputLayer((10, None, 4, 4))
        with pytest.raises(ValueError) as exc:
            LocallyConnected2DLayer(input_layer, 4, 3, channelwise=True)
        assert "A LocallyConnected2DLayer requires a fixed input shape " \
               "(except for the batch size)" in exc.value.args[0]
