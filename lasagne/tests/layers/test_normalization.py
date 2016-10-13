# -*- coding: utf-8 -*-

"""

The :func:`ground_truth_normalizer()`, :func:`ground_truth_normalize_row` and
:class:`TestLocalResponseNormalization2DLayer` implementations contain code
from `pylearn2 <http://github.com/lisa-lab/pylearn2>`_, which is covered
by the following license:


Copyright (c) 2011--2014, Université de Montréal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from mock import Mock
import numpy as np
import pytest
import theano


def ground_truth_normalizer(c01b, k, n, alpha, beta):
    out = np.zeros(c01b.shape)

    for r in range(out.shape[1]):
        for c in range(out.shape[2]):
            for x in range(out.shape[3]):
                out[:, r, c, x] = ground_truth_normalize_row(
                        row=c01b[:, r, c, x],
                        k=k, n=n, alpha=alpha, beta=beta)
    return out


def ground_truth_normalize_row(row, k, n, alpha, beta):
    assert row.ndim == 1
    out = np.zeros(row.shape)
    for i in range(row.shape[0]):
        s = k
        tot = 0
        for j in range(max(0, i-n//2), min(row.shape[0], i+n//2+1)):
            tot += 1
            sq = row[j] ** 2.
            assert sq > 0.
            assert s >= k
            assert alpha > 0.
            s += alpha * sq
            assert s >= k
        assert tot <= n
        assert s >= k
        s = s ** beta
        out[i] = row[i] / s
    return out


class TestLocalResponseNormalization2DLayer:

    @pytest.fixture
    def rng(self):
        return np.random.RandomState([2013, 2])

    @pytest.fixture
    def input_data(self, rng):
        channels = 15
        rows = 3
        cols = 4
        batch_size = 2
        shape = (batch_size, channels, rows, cols)
        return rng.randn(*shape).astype(theano.config.floatX)

    @pytest.fixture
    def input_layer(self, input_data):
        from lasagne.layers.input import InputLayer
        shape = list(input_data.shape)
        shape[0] = None
        return InputLayer(shape)

    @pytest.fixture
    def layer(self, input_layer):

        from lasagne.layers.normalization import\
                LocalResponseNormalization2DLayer

        layer = LocalResponseNormalization2DLayer(input_layer,
                                                  alpha=1.5,
                                                  k=2,
                                                  beta=0.75,
                                                  n=5)
        return layer

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for((1, 2, 3, 4)) == (1, 2, 3, 4)

    def test_even_n_fails(self, input_layer):
        from lasagne.layers.normalization import\
                LocalResponseNormalization2DLayer

        with pytest.raises(NotImplementedError):
            LocalResponseNormalization2DLayer(input_layer, n=4)

    def test_normalization(self, input_data, input_layer, layer):
        from lasagne.layers import get_output
        X = input_layer.input_var
        lrn = theano.function([X], get_output(layer, X))
        out = lrn(input_data)

        # ground_truth_normalizer assumes c01b
        input_data_c01b = input_data.transpose([1, 2, 3, 0])
        ground_out = ground_truth_normalizer(input_data_c01b,
                                             n=layer.n, k=layer.k,
                                             alpha=layer.alpha,
                                             beta=layer.beta)
        ground_out = np.transpose(ground_out, [3, 0, 1, 2])

        assert out.shape == ground_out.shape

        assert np.allclose(out, ground_out)


class TestBatchNormLayer:
    @pytest.fixture(params=(False, True), ids=('plain', 'dnn'))
    def BatchNormLayer(self, request):
        dnn = request.param
        if not dnn:
            from lasagne.layers.normalization import BatchNormLayer
        elif dnn:
            try:
                from lasagne.layers.dnn import (
                        BatchNormDNNLayer as BatchNormLayer)
            except ImportError:
                pytest.skip("cuDNN batch norm not available")
        return BatchNormLayer

    @pytest.fixture
    def init_unique(self):
        # initializer for a tensor of unique values
        return lambda shape: np.arange(np.prod(shape)).reshape(shape)

    def test_init(self, BatchNormLayer, init_unique):
        input_shape = (2, 3, 4)
        # default: normalize over all but second axis
        beta = BatchNormLayer(input_shape, beta=init_unique).beta
        assert np.allclose(beta.get_value(), init_unique((3,)))
        # normalize over first axis only
        beta = BatchNormLayer(input_shape, beta=init_unique, axes=0).beta
        assert np.allclose(beta.get_value(), init_unique((3, 4)))
        # normalize over second and third axis
        try:
            beta = BatchNormLayer(
                    input_shape, beta=init_unique, axes=(1, 2)).beta
            assert np.allclose(beta.get_value(), init_unique((2,)))
        except ValueError as exc:
            assert "BatchNormDNNLayer only supports" in exc.args[0]

    @pytest.mark.parametrize('update_averages', [None, True, False])
    @pytest.mark.parametrize('use_averages', [None, True, False])
    @pytest.mark.parametrize('deterministic', [True, False])
    def test_get_output_for(self, BatchNormLayer, deterministic, use_averages,
                            update_averages):
        input_shape = (20, 30, 40)

        # random input tensor, beta, gamma, mean, inv_std and alpha
        input = (np.random.randn(*input_shape).astype(theano.config.floatX) +
                 np.random.randn(1, 30, 1).astype(theano.config.floatX))
        beta = np.random.randn(30).astype(theano.config.floatX)
        gamma = np.random.randn(30).astype(theano.config.floatX)
        mean = np.random.randn(30).astype(theano.config.floatX)
        inv_std = np.random.rand(30).astype(theano.config.floatX)
        alpha = np.random.rand()

        # create layer (with default axes: normalize over all but second axis)
        layer = BatchNormLayer(input_shape, beta=beta, gamma=gamma, mean=mean,
                               inv_std=inv_std, alpha=alpha)

        # call get_output_for()
        kwargs = {'deterministic': deterministic}
        if use_averages is not None:
            kwargs['batch_norm_use_averages'] = use_averages
        else:
            use_averages = deterministic
        if update_averages is not None:
            kwargs['batch_norm_update_averages'] = update_averages
        else:
            update_averages = not deterministic
        result = layer.get_output_for(theano.tensor.constant(input),
                                      **kwargs).eval()

        # compute expected results and expected updated parameters
        input_mean = input.mean(axis=(0, 2))
        input_inv_std = 1 / np.sqrt(input.var(axis=(0, 2)) + layer.epsilon)
        if use_averages:
            use_mean, use_inv_std = mean, inv_std
        else:
            use_mean, use_inv_std = input_mean, input_inv_std
        bcast = (np.newaxis, slice(None), np.newaxis)
        exp_result = (input - use_mean[bcast]) * use_inv_std[bcast]
        exp_result = exp_result * gamma[bcast] + beta[bcast]
        if update_averages:
            new_mean = (1 - alpha) * mean + alpha * input_mean
            new_inv_std = (1 - alpha) * inv_std + alpha * input_inv_std
        else:
            new_mean, new_inv_std = mean, inv_std

        # compare expected results to actual results
        tol = {'atol': 1e-5, 'rtol': 1e-6}
        assert np.allclose(layer.mean.get_value(), new_mean, **tol)
        assert np.allclose(layer.inv_std.get_value(), new_inv_std, **tol)
        assert np.allclose(result, exp_result, **tol)

    def test_undefined_shape(self, BatchNormLayer):
        # should work:
        BatchNormLayer((64, 2, None), axes=(0, 2))
        # should not work:
        with pytest.raises(ValueError) as exc:
            BatchNormLayer((64, None, 3), axes=(0, 2))
        assert 'needs specified input sizes' in exc.value.args[0]

    def test_skip_linear_transform(self, BatchNormLayer):
        input_shape = (20, 30, 40)

        # random input tensor, beta, gamma
        input = (np.random.randn(*input_shape).astype(theano.config.floatX) +
                 np.random.randn(1, 30, 1).astype(theano.config.floatX))
        beta = np.random.randn(30).astype(theano.config.floatX)
        gamma = np.random.randn(30).astype(theano.config.floatX)

        # create layers without beta or gamma
        layer1 = BatchNormLayer(input_shape, beta=None, gamma=gamma)
        layer2 = BatchNormLayer(input_shape, beta=beta, gamma=None)

        # check that one parameter is missing
        assert len(layer1.get_params()) == 3
        assert len(layer2.get_params()) == 3

        # call get_output_for()
        result1 = layer1.get_output_for(theano.tensor.constant(input),
                                        deterministic=False).eval()
        result2 = layer2.get_output_for(theano.tensor.constant(input),
                                        deterministic=False).eval()

        # compute expected results and expected updated parameters
        mean = input.mean(axis=(0, 2))
        std = np.sqrt(input.var(axis=(0, 2)) + layer1.epsilon)
        exp_result = (input - mean[None, :, None]) / std[None, :, None]
        exp_result1 = exp_result * gamma[None, :, None]  # no beta
        exp_result2 = exp_result + beta[None, :, None]  # no gamma

        # compare expected results to actual results
        tol = {'atol': 1e-5, 'rtol': 1e-6}
        assert np.allclose(result1, exp_result1, **tol)
        assert np.allclose(result2, exp_result2, **tol)


@pytest.mark.parametrize('dnn', [False, True])
def test_batch_norm_macro(dnn):
    if not dnn:
        from lasagne.layers import (BatchNormLayer, batch_norm)
    else:
        try:
            from lasagne.layers.dnn import (
                    BatchNormDNNLayer as BatchNormLayer,
                    batch_norm_dnn as batch_norm)
        except ImportError:
            pytest.skip("cuDNN batch norm not available")
    from lasagne.layers import (Layer, NonlinearityLayer)
    from lasagne.nonlinearities import identity
    input_shape = (2, 3)
    obj = object()

    # check if it steals the nonlinearity
    layer = Mock(Layer, output_shape=input_shape, nonlinearity=obj)
    bnstack = batch_norm(layer)
    assert isinstance(bnstack, NonlinearityLayer)
    assert isinstance(bnstack.input_layer, BatchNormLayer)
    assert layer.nonlinearity is identity
    assert bnstack.nonlinearity is obj

    # check if it removes the bias
    layer = Mock(Layer, output_shape=input_shape, b=obj, params={obj: set()})
    bnstack = batch_norm(layer)
    assert isinstance(bnstack, BatchNormLayer)
    assert layer.b is None
    assert obj not in layer.params

    # check if it can handle an unset bias
    layer = Mock(Layer, output_shape=input_shape, b=None, params={obj: set()})
    bnstack = batch_norm(layer)
    assert isinstance(bnstack, BatchNormLayer)
    assert layer.b is None

    # check if it passes on kwargs
    layer = Mock(Layer, output_shape=input_shape)
    bnstack = batch_norm(layer, name='foo')
    assert isinstance(bnstack, BatchNormLayer)
    assert bnstack.name == 'foo'

    # check if created layers are named with kwargs name
    layer = Mock(Layer, output_shape=input_shape, nonlinearity=obj)
    layer.name = 'foo'
    bnstack = batch_norm(layer, name='foo_bnorm')
    assert isinstance(bnstack, NonlinearityLayer)
    assert isinstance(bnstack.input_layer, BatchNormLayer)
    assert bnstack.name == 'foo_bnorm_nonlin'
    assert bnstack.input_layer.name == 'foo_bnorm'

    # check if created layers are named with wrapped layer name
    layer = Mock(Layer, output_shape=input_shape, nonlinearity=obj)
    layer.name = 'foo'
    bnstack = batch_norm(layer)
    assert isinstance(bnstack, NonlinearityLayer)
    assert isinstance(bnstack.input_layer, BatchNormLayer)
    assert bnstack.name == 'foo_bn_nonlin'
    assert bnstack.input_layer.name == 'foo_bn'

    # check if created layers remain unnamed if no names are given
    layer = Mock(Layer, output_shape=input_shape, nonlinearity=obj)
    bnstack = batch_norm(layer)
    assert isinstance(bnstack, NonlinearityLayer)
    assert isinstance(bnstack.input_layer, BatchNormLayer)
    assert bnstack.name is None
    assert bnstack.input_layer.name is None
