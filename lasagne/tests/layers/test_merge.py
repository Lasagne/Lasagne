from mock import Mock
import numpy
import pytest
import theano


class TestConcatLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.merge import ConcatLayer
        return ConcatLayer([Mock(), Mock()], axis=1)

    def test_get_output_shape_for(self, layer):
        input_shapes = [(3, 2), (3, 5)]
        result = layer.get_output_shape_for(input_shapes)
        assert result == (3, 7)

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

    def test_bad_coeffs_fails(self, layer):
        from lasagne.layers.merge import ElemwiseSumLayer
        with pytest.raises(ValueError):
            ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, 3, -1])

    def test_get_output_shape_for_fails(self, layer):
        input_shapes = [(3, 2), (3, 5)]
        with pytest.raises(ValueError):
            layer.get_output_shape_for(input_shapes)


class TestElemwiseMergeLayerMul:
    @pytest.fixture
    def layer(self):
        import theano.tensor as T
        from lasagne.layers.merge import ElemwiseMergeLayer
        return ElemwiseMergeLayer([Mock(), Mock()], merge_function=T.mul)

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = a*b
        assert (result_eval == desired_result).all()


class TestElemwiseMergeLayerMaximum:
    @pytest.fixture
    def layer(self):
        import theano.tensor as T
        from lasagne.layers.merge import ElemwiseMergeLayer
        return ElemwiseMergeLayer([Mock(), Mock()], merge_function=T.maximum)

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = numpy.maximum(a, b)
        assert (result_eval == desired_result).all()
