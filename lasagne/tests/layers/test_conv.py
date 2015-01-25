from mock import Mock
import numpy as np
import pytest
import theano
from theano.tensor.nnet import conv2d

from lasagne.utils import floatX


def conv2d_test_sets():
    def _convert(input, kernel, output, kwargs):
        return [theano.shared(floatX(input)), floatX(kernel), output, kwargs]

    input = np.random.random((3, 1, 16, 16))
    kernel = np.random.random((16, 1, 3, 3))
    output = conv2d(input, kernel).eval()
    yield _convert(input, kernel, output, {})

    input = np.random.random((3, 3, 16, 16))
    kernel = np.random.random((16, 3, 3, 3))
    output = conv2d(input, kernel).eval()
    yield _convert(input, kernel, output, {})


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
            ('Conv2DLayer', {}),
            ('Conv2DCCLayer', {'flip_filters': True}),
            ('Conv2DMMLayer', {'flip_filters': True}),
            ('Conv2DDNNLayer', {'flip_filters': True}),
            ],
        )
    def Conv2DImpl(self, request):
        from lasagne import layers
        impl_name, impl_default_kwargs = request.param
        try:
            impl = getattr(layers, impl_name)
        except AttributeError:
            pytest.skip("{} not available".format(request.param))

        def wrapper(*args, **kwargs):
            kwargs2 = impl_default_kwargs.copy()
            kwargs2.update(kwargs)
            return impl(*args, **kwargs2)

        wrapper.__name__ = impl_name
        return wrapper

    @pytest.mark.parametrize(
        "input, kernel, output, kwargs", list(conv2d_test_sets()))
    @pytest.mark.parametrize("extra_kwargs", [
        {},
        {'untie_biases': True},
        ])
    def test_defaults(self, Conv2DImpl, DummyInputLayer,
                      input, kernel, output, kwargs, extra_kwargs):
        kwargs.update(extra_kwargs)
        input_layer = DummyInputLayer(input.shape.eval())
        layer = Conv2DImpl(
            input_layer,
            num_filters=kernel.shape[0],
            filter_size=kernel.shape[2:],
            W=kernel,
            **kwargs
            )
        actual = layer.get_output(input).eval()
        assert actual.shape == output.shape
        assert np.allclose(actual, output)
