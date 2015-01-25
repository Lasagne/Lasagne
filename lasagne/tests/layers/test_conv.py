from mock import Mock
import numpy as np
import pytest

from lasagne.utils import floatX


def input_kernel_and_output_one_sample():
    input = np.array([[[
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
        ]]])
    kernel = np.array([[[
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        ]]])
    output = np.array([[[
        [4, 3, 4],
        [2, 4, 3],
        [2, 3, 4],
        ]]])
    return map(floatX, [input, kernel, output])


def input_kernel_and_output_multiple_samples():
    input, kernel, output = input_kernel_and_output_one_sample()

    input = np.repeat(input, 5, axis=0)
    output = np.repeat(output, 5, axis=0)

    return map(floatX, [input, kernel, output])


def input_kernel_and_output_multiple_filters():
    input, kernel, output = input_kernel_and_output_one_sample()

    kernel = np.array([
        [[
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]],
        [[
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]],
    ])

    output = np.array([[
        [
            [4, 3, 4],
            [2, 4, 3],
            [2, 3, 4],
        ],
        [
            [2, 3, 2],
            [1, 2, 3],
            [1, 2, 2],
        ],
        ]])

    return map(floatX, [input, kernel, output])


def input_kernel_and_output_multiple_channels():
    input, kernel, output = input_kernel_and_output_multiple_filters()

    # all the inputs channels are identical:
    input = np.repeat(input, 2, axis=1)
    kernel = kernel.transpose(1, 0, 2, 3)
    output = output.sum(axis=1)[None, :]
    return map(floatX, [input, kernel, output])


@pytest.fixture
def DummyInputLayer():
    def factory(get_output_shape):
        return Mock(
            get_output_shape=lambda: get_output_shape,
            get_output=lambda input: input,
            )
    return factory


class TestConv2DLayerImplementations:
    @pytest.fixture(
        params=[
            "Conv2DLayer",
            "Conv2DMMLayer",
            "Conv2DDNNLayer",  # untested
            ],
        )
    def Conv2DImpl(self, request):
        from lasagne import layers
        try:
            return getattr(layers, request.param)
        except AttributeError:
            pytest.skip("{} not available".format(request.param))

    @pytest.mark.parametrize("input, kernel, output", [
        input_kernel_and_output_one_sample(),
        input_kernel_and_output_multiple_samples(),
        input_kernel_and_output_multiple_filters(),
        input_kernel_and_output_multiple_channels(),
        ])
    @pytest.mark.parametrize("kwargs", [
        {},
        {'untie_biases': True},
        ])
    def test_defaults(self, Conv2DImpl, DummyInputLayer,
                      input, kernel, output, kwargs):
        input_layer = DummyInputLayer(input.shape)
        layer = Conv2DImpl(
            input_layer,
            num_filters=kernel.shape[0],
            filter_size=kernel.shape[2:],
            W=kernel,
            **kwargs
            )
        actual = layer.get_output(input).eval()
        assert actual.shape == output.shape
        assert (actual == output).all()
