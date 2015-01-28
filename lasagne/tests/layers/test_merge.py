from __future__ import absolute_import
from mock import Mock
import numpy
import pytest
import theano


class TestConcatLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.merge import ConcatLayer
        return ConcatLayer([Mock(), Mock()], axis=1)

    def test_get_output_for(self, layer):
        inputs = [theano.shared(numpy.ones((3, 3))),
            theano.shared(numpy.ones((3, 2)))]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = numpy.hstack([input.get_value() for input in inputs])
        assert (result_eval == desired_result).all()


class TestElemwiseSumLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.merge import ElemwiseSumLayer
        return ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, -1])

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = 2*a - b
        assert (result_eval == desired_result).all()
