from mock import Mock
import numpy as np
import pytest
import importlib
import theano

import lasagne


class TestLocalResponseNormalization2DLayer:

    @pytest.fixture
    def layer(self, dummy_input_layer):
        from lasagne.layers.normalization import\
                LocalResponseNormalization2DLayer

        layer = LocalResponseNormalization2DLayer(dummy_input_layer,
                                                  alpha=1.5,
                                                  k=2,
                                                  beta=0.75,
                                                  n=0.75)

        return layer

    def test_get_params(self, layer):
        assert len(layer.get_params()) == 0

    def test_get_bias_params(self, layer):
        assert len(layer.get_bias_params()) == 0
