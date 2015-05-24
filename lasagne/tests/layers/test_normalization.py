# -*- coding: utf-8 -*-

"""

This file contains code from pylearn2, which is covered by the following
license:


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


import numpy as np
import pytest
import theano

import lasagne


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
        X = input_layer.input_var
        lrn = theano.function([X], lasagne.layers.get_output(layer, X))
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
