from mock import Mock
import numpy
import pytest
import theano


class TestAutocrop:
    # Test internal helper methods of MergeCropLayer
    def test_autocrop_array_shapes(self):
        from lasagne.layers.merge import autocrop_array_shapes
        crop0 = None
        crop1 = [None, 'lower', 'center', 'upper']
        # Too few crop modes; should get padded with None
        crop2 = ['lower', 'upper']
        # Invalid crop modes
        crop_bad = ['lower', 'upper', 'bad', 'worse']

        assert autocrop_array_shapes(
            [(1, 2, 3, 4), (5, 6, 7, 8), (5, 4, 3, 2)], crop0) == \
            [(1, 2, 3, 4), (5, 6, 7, 8), (5, 4, 3, 2)]
        assert autocrop_array_shapes(
            [(1, 2, 3, 4), (5, 6, 7, 8), (5, 4, 3, 2)], crop1) == \
            [(1, 2, 3, 2), (5, 2, 3, 2), (5, 2, 3, 2)]
        assert autocrop_array_shapes(
            [(1, 2, 3, 4), (5, 6, 7, 8), (5, 4, 3, 2)], crop2) == \
            [(1, 2, 3, 4), (1, 2, 7, 8), (1, 2, 3, 2)]

        with pytest.raises(ValueError):
            autocrop_array_shapes(
                [(1, 2, 3, 4), (5, 6, 7, 8), (5, 4, 3, 2)], crop_bad)

        # Inconsistent dimensionality
        with pytest.raises(ValueError):
            autocrop_array_shapes(
                [(1, 2, 3, 4), (5, 6, 7), (5, 4, 3, 2, 10)], crop1)

    def test_crop_inputs(self):
        from lasagne.layers.merge import autocrop
        from numpy.testing import assert_array_equal
        crop_0 = None
        crop_1 = [None, 'lower', 'center', 'upper']
        crop_l = ['lower', 'lower', 'lower', 'lower']
        crop_c = ['center', 'center', 'center', 'center']
        crop_u = ['upper', 'upper', 'upper', 'upper']
        crop_x = ['lower', 'lower']
        crop_bad = ['lower', 'lower', 'bad', 'worse']

        x0 = numpy.random.random((2, 3, 5, 7))
        x1 = numpy.random.random((1, 2, 3, 4))
        x2 = numpy.random.random((6, 3, 4, 2))

        def crop_test(cropping, inputs, expected):
            inputs = [theano.shared(x) for x in inputs]
            outs = autocrop(inputs, cropping)
            outs = [o.eval() for o in outs]
            assert len(outs) == len(expected)
            for o, e in zip(outs, expected):
                assert_array_equal(o, e)

        crop_test(crop_0, [x0, x1],
                  [x0, x1])
        crop_test(crop_1, [x0, x1],
                  [x0[:, :2, 1:4, 3:], x1[:, :, :, :]])
        crop_test(crop_l, [x0, x1],
                  [x0[:1, :2, :3, :4], x1[:, :, :, :]])
        crop_test(crop_c, [x0, x1],
                  [x0[:1, :2, 1:4, 1:5], x1[:, :, :, :]])
        crop_test(crop_u, [x0, x1],
                  [x0[1:, 1:, 2:, 3:], x1[:, :, :, :]])

        crop_test(crop_0, [x0, x2],
                  [x0, x2])
        crop_test(crop_1, [x0, x2],
                  [x0[:, :, :4, 5:], x2[:, :, :, :]])
        crop_test(crop_l, [x0, x2],
                  [x0[:, :, :4, :2], x2[:2, :, :, :]])
        crop_test(crop_c, [x0, x2],
                  [x0[:, :, :4, 2:4], x2[2:4, :, :, :]])
        crop_test(crop_u, [x0, x2],
                  [x0[:, :, 1:, 5:], x2[4:, :, :, :]])

        crop_test(crop_0, [x0, x1, x2],
                  [x0, x1, x2])
        crop_test(crop_1, [x0, x1, x2],
                  [x0[:, :2, 1:4, 5:], x1[:, :, :, 2:], x2[:, :2, :3, :]])
        crop_test(crop_l, [x0, x1, x2],
                  [x0[:1, :2, :3, :2], x1[:, :, :, :2], x2[:1, :2, :3, :]])
        crop_test(crop_c, [x0, x1, x2],
                  [x0[:1, :2, 1:4, 2:4], x1[:, :, :, 1:3], x2[2:3, :2, :3, :]])
        crop_test(crop_u, [x0, x1, x2],
                  [x0[1:, 1:, 2:, 5:], x1[:, :, :, 2:], x2[5:, 1:, 1:, :]])

        crop_test(crop_x, [x0, x1, x2],
                  [x0[:1, :2, :, :], x1[:1, :2, :, :], x2[:1, :2, :, :]])

        # test that num outputs is correct when the number of inputs is
        # larger than ndim of the inputs.
        crop_test(crop_x, [x0, x1, x2, x0, x1, x2],
                  [x0[:1, :2, :, :], x1[:1, :2, :, :], x2[:1, :2, :, :],
                   x0[:1, :2, :, :], x1[:1, :2, :, :], x2[:1, :2, :, :]])

        with pytest.raises(ValueError):
            crop_test(crop_bad, [x0, x1, x2],
                      [x0[:1, :2, :, :], x1[:1, :2, :, :], x2[:1, :2, :, :]])

        # Inconsistent dimensionality
        with pytest.raises(ValueError):
            crop_test(crop_bad, [x0[:, :, :, 0], x1, x2[:, :, :, :, None]],
                      [x0[:1, :2, :, :], x1[:1, :2, :, :], x2[:1, :2, :, :]])


class TestConcatLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.merge import ConcatLayer
        return ConcatLayer([Mock(), Mock()], axis=1)

    @pytest.fixture
    def crop_layer_0(self):
        from lasagne.layers.merge import ConcatLayer
        return ConcatLayer([Mock(), Mock()], axis=0,
                           cropping=['lower'] * 2)

    @pytest.fixture
    def crop_layer_1(self):
        from lasagne.layers.merge import ConcatLayer
        return ConcatLayer([Mock(), Mock()], axis=1,
                           cropping=['lower'] * 2)

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for([(3, 2), (3, 5)]) == (3, 7)
        assert layer.get_output_shape_for([(3, 2), (3, None)]) == (3, None)
        assert layer.get_output_shape_for([(None, 2), (3, 5)]) == (3, 7)
        assert layer.get_output_shape_for([(None, 2), (None, 5)]) == (None, 7)
        with pytest.raises(ValueError):
            layer.get_output_shape_for([(4, None), (3, 5)])
        with pytest.raises(ValueError):
            layer.get_output_shape_for([(3, 2), (4, None)])
        with pytest.raises(ValueError):
            layer.get_output_shape_for([(None, 2), (3, 5), (4, 5)])

    def test_get_output_shape_for_cropped(self, crop_layer_0, crop_layer_1):
        input_shapes = [(3, 2), (4, 5)]
        result_0 = crop_layer_0.get_output_shape_for(input_shapes)
        result_1 = crop_layer_1.get_output_shape_for(input_shapes)
        assert result_0 == (7, 2)
        assert result_1 == (3, 7)

    def test_get_output_for(self, layer):
        inputs = [theano.shared(numpy.ones((3, 3))),
                  theano.shared(numpy.ones((3, 2)))]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = numpy.hstack([input.get_value() for input in inputs])
        assert (result_eval == desired_result).all()

    def test_get_output_for_cropped(self, crop_layer_0, crop_layer_1):
        x0 = numpy.random.random((5, 3))
        x1 = numpy.random.random((4, 2))
        inputs = [theano.shared(x0),
                  theano.shared(x1)]
        result_0 = crop_layer_0.get_output_for(inputs).eval()
        result_1 = crop_layer_1.get_output_for(inputs).eval()
        desired_result_0 = numpy.concatenate([x0[:, :2], x1[:, :2]], axis=0)
        desired_result_1 = numpy.concatenate([x0[:4, :], x1[:4, :]], axis=1)
        assert (result_0 == desired_result_0).all()
        assert (result_1 == desired_result_1).all()


class TestElemwiseSumLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.merge import ElemwiseSumLayer
        return ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, -1])

    @pytest.fixture
    def crop_layer(self):
        from lasagne.layers.merge import ElemwiseSumLayer
        return ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, -1],
                                cropping=['lower'] * 2)

    def test_get_output_shape_for(self, layer):
        assert layer.get_output_shape_for([(3, 2), (3, 2)]) == (3, 2)
        assert layer.get_output_shape_for([(3, 2), (3, None)]) == (3, 2)
        assert layer.get_output_shape_for([(None, 2), (3, 2)]) == (3, 2)
        assert layer.get_output_shape_for([(None, 2), (None, 2)]) == (None, 2)
        with pytest.raises(ValueError):
            layer.get_output_shape_for([(3, None), (4, 2)])
        with pytest.raises(ValueError):
            layer.get_output_shape_for([(3, 2), (4, None)])
        with pytest.raises(ValueError):
            layer.get_output_shape_for([(None, 2), (3, 2), (4, 2)])

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = 2*a - b
        assert (result_eval == desired_result).all()

    def test_get_output_for_cropped(self, crop_layer):
        from numpy.testing import assert_array_almost_equal as aeq
        x0 = numpy.random.random((5, 3))
        x1 = numpy.random.random((4, 2))
        inputs = [theano.shared(x0),
                  theano.shared(x1)]
        result = crop_layer.get_output_for(inputs).eval()
        desired_result = 2*x0[:4, :2] - x1[:4, :2]
        aeq(result, desired_result)

    def test_bad_coeffs_fails(self, layer):
        from lasagne.layers.merge import ElemwiseSumLayer
        with pytest.raises(ValueError):
            ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, 3, -1])


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
