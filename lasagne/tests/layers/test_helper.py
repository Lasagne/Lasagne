from mock import Mock, PropertyMock
import pytest
import numpy
import theano


class TestGetAllLayers:
    def test_stack(self):
        from lasagne.layers import InputLayer, DenseLayer, get_all_layers
        from itertools import permutations
        # l1 --> l2 --> l3
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)
        # try all possible combinations and orders for a query
        for count in (0, 1, 2, 3):
            for query in permutations([l1, l2, l3], count):
                if l3 in query:
                    expected = [l1, l2, l3]
                elif l2 in query:
                    expected = [l1, l2]
                elif l1 in query:
                    expected = [l1]
                else:
                    expected = []
                assert get_all_layers(query) == expected
        # treat_as_input=[l2] should block l1 from appearing
        assert get_all_layers(l3, treat_as_input=[l2]) == [l2, l3]

    def test_merge(self):
        from lasagne.layers import (InputLayer, DenseLayer, ElemwiseSumLayer,
                                    get_all_layers)
        # l1 --> l2 --> l3 --> l6
        #        l4 --> l5 ----^
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)
        l4 = InputLayer((10, 30))
        l5 = DenseLayer(l4, 40)
        l6 = ElemwiseSumLayer([l3, l5])
        # try various combinations and orders for a query
        assert get_all_layers(l6) == [l1, l2, l3, l4, l5, l6]
        assert get_all_layers([l4, l6]) == [l4, l1, l2, l3, l5, l6]
        assert get_all_layers([l5, l6]) == [l4, l5, l1, l2, l3, l6]
        assert get_all_layers([l4, l2, l5, l6]) == [l4, l1, l2, l5, l3, l6]
        # check that treat_as_input correctly blocks the search
        assert get_all_layers(l6, treat_as_input=[l2]) == [l2, l3, l4, l5, l6]
        assert get_all_layers(l6, treat_as_input=[l3, l5]) == [l3, l5, l6]
        assert get_all_layers([l6, l2], treat_as_input=[l6]) == [l6, l1, l2]

    def test_split(self):
        from lasagne.layers import InputLayer, DenseLayer, get_all_layers
        # l1 --> l2 --> l3
        #  \---> l4
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)
        l4 = DenseLayer(l1, 50)
        # try various combinations and orders for a query
        assert get_all_layers(l3) == [l1, l2, l3]
        assert get_all_layers(l4) == [l1, l4]
        assert get_all_layers([l3, l4]) == [l1, l2, l3, l4]
        assert get_all_layers([l4, l3]) == [l1, l4, l2, l3]
        # check that treat_as_input correctly blocks the search
        assert get_all_layers(l3, treat_as_input=[l2]) == [l2, l3]
        assert get_all_layers([l3, l4], treat_as_input=[l2]) == [l2, l3,
                                                                 l1, l4]

    def test_bridge(self):
        from lasagne.layers import (InputLayer, DenseLayer, ElemwiseSumLayer,
                                    get_all_layers)
        # l1 --> l2 --> l3 --> l4 --> l5
        #         \------------^
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 30)
        l4 = ElemwiseSumLayer([l2, l3])
        l5 = DenseLayer(l4, 40)
        # check for correct topological order
        assert get_all_layers(l5) == [l1, l2, l3, l4, l5]
        # check that treat_as_input=[l4] blocks the search and =[l3] does not
        assert get_all_layers(l5, treat_as_input=[l4]) == [l4, l5]
        assert get_all_layers(l5, treat_as_input=[l3]) == [l1, l2, l3, l4, l5]


class TestGetOutput_InputLayer:
    @pytest.fixture
    def get_output(self):
        from lasagne.layers.helper import get_output
        return get_output

    @pytest.fixture
    def layer(self):
        from lasagne.layers.input import InputLayer
        return InputLayer((3, 2))

    def test_get_output_without_arguments(self, layer, get_output):
        assert get_output(layer) is layer.input_var

    def test_get_output_input_is_variable(self, layer, get_output):
        variable = theano.Variable("myvariable")
        assert get_output(layer, variable) is variable

    def test_get_output_input_is_array(self, layer, get_output):
        inputs = [[1, 2, 3]]
        output = get_output(layer, inputs)
        assert numpy.all(output.eval() == inputs)

    def test_get_output_input_is_a_mapping(self, layer, get_output):
        inputs = {layer: theano.tensor.matrix()}
        assert get_output(layer, inputs) is inputs[layer]


class TestGetOutput_Layer:
    @pytest.fixture
    def get_output(self):
        from lasagne.layers.helper import get_output
        return get_output

    @pytest.fixture
    def layers(self):
        from lasagne.layers.base import Layer
        from lasagne.layers.input import InputLayer
        # create a mock that has the same attributes as an InputLayer instance
        l1 = Mock(InputLayer((None,)))
        # create a mock that has the same attributes as a Layer instance
        l2 = Mock(Layer(l1))
        # link it to the InputLayer mock
        l2.input_layer = l1
        # create another mock that has the same attributes as a Layer instance
        l3 = Mock(Layer(l2))
        # link it to the first mock, to get an "l1 --> l2 --> l3" chain
        l3.input_layer = l2
        return l1, l2, l3

    def test_get_output_without_arguments(self, layers, get_output):
        l1, l2, l3 = layers
        output = get_output(l3)
        # expected: l3.get_output_for(l2.get_output_for(l1.input_var))
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with(
            l2.get_output_for.return_value)
        l2.get_output_for.assert_called_with(
            l1.input_var)

    def test_get_output_with_single_argument(self, layers, get_output):
        l1, l2, l3 = layers
        inputs, kwarg = theano.tensor.matrix(), object()
        output = get_output(l3, inputs, kwarg=kwarg)
        # expected: l3.get_output_for(l2.get_output_for(inputs, kwarg=kwarg),
        #                             kwarg=kwarg)
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with(
            l2.get_output_for.return_value, kwarg=kwarg)
        l2.get_output_for.assert_called_with(
            inputs, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layers, get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1).input_var = p
        inputs = {l3: theano.tensor.matrix()}
        # expected: inputs[l3]
        assert get_output(l3, inputs) is inputs[l3]
        # l3.get_output_for, l2.get_output_for should not have been called
        assert l3.get_output_for.call_count == 0
        assert l2.get_output_for.call_count == 0
        # l1.input_var should not have been accessed
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_no_key(self, layers, get_output):
        l1, l2, l3 = layers
        output = get_output(l3, {})
        # expected: l3.get_output_for(l2.get_output_for(l1.input_var))
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with(
            l2.get_output_for.return_value)
        l2.get_output_for.assert_called_with(
            l1.input_var)

    def test_get_output_input_is_a_mapping_to_array(self, layers, get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1).input_var = p
        inputs = {l3: [[1, 2, 3]]}
        output = get_output(l3, inputs)
        # expected: inputs[l3]
        assert numpy.all(output.eval() == inputs[l3])
        # l3.get_output_for, l2.get_output_for should not have been called
        assert l3.get_output_for.call_count == 0
        assert l2.get_output_for.call_count == 0
        # l1.input_var should not have been accessed
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_layer(self, layers, get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1).input_var = p
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {l2: input_expr}
        output = get_output(l3, inputs, kwarg=kwarg)
        # expected: l3.get_output_for(input_expr, kwarg=kwarg)
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with(input_expr, kwarg=kwarg)
        # l2.get_output_for should not have been called
        assert l2.get_output_for.call_count == 0
        # l1.input_var should not have been accessed
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_input_layer(self, layers,
                                                           get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1).input_var = p
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {l1: input_expr}
        output = get_output(l3, inputs, kwarg=kwarg)
        # expected: l3.get_output_for(l2.get_output_for(input_expr,
        #                                               kwarg=kwarg),
        #                             kwarg=kwarg)
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with(
            l2.get_output_for.return_value, kwarg=kwarg)
        l2.get_output_for.assert_called_with(
            input_expr, kwarg=kwarg)
        # l1.input_var should not have been accessed
        assert p.call_count == 0

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import Layer
        return Layer((None, 20))

    def test_layer_from_shape_invalid_get_output(self, layer_from_shape,
                                                 get_output):
        layer = layer_from_shape
        with pytest.raises(ValueError):
            get_output(layer)
        with pytest.raises(ValueError):
            get_output(layer, [1, 2])
        with pytest.raises(ValueError):
            get_output(layer, {Mock(): [1, 2]})

    def test_layer_from_shape_valid_get_output(self, layer_from_shape,
                                               get_output):
        layer = layer_from_shape
        inputs = {layer: theano.tensor.matrix()}
        assert get_output(layer, inputs) is inputs[layer]
        inputs = {None: theano.tensor.matrix()}
        layer.get_output_for = Mock()
        assert get_output(layer, inputs) is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(inputs[None])


class TestGetOutput_MergeLayer:
    @pytest.fixture
    def get_output(self):
        from lasagne.layers.helper import get_output
        return get_output

    @pytest.fixture
    def layers(self):
        from lasagne.layers.base import Layer, MergeLayer
        from lasagne.layers.input import InputLayer
        # create two mocks of the same attributes as an InputLayer instance
        l1 = [Mock(InputLayer((None,))), Mock(InputLayer((None,)))]
        # create two mocks of the same attributes as a Layer instance
        l2 = [Mock(Layer(l1[0])), Mock(Layer(l1[1]))]
        # link them to the InputLayer mocks
        l2[0].input_layer = l1[0]
        l2[1].input_layer = l1[1]
        # create a mock that has the same attributes as a MergeLayer
        l3 = Mock(MergeLayer(l2))
        # link it to the two layer mocks, to get the following network:
        # l1[0] --> l2[0] --> l3
        # l1[1] --> l2[1] ----^
        l3.input_layers = l2
        return l1, l2, l3

    def test_get_output_without_arguments(self, layers, get_output):
        l1, l2, l3 = layers
        output = get_output(l3)
        # expected: l3.get_output_for([l2[0].get_output_for(l1[0].input_var),
        #                              l2[1].get_output_for(l1[1].input_var)])
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with([
            l2[0].get_output_for.return_value,
            l2[1].get_output_for.return_value,
            ])
        l2[0].get_output_for.assert_called_with(
            l1[0].input_var)
        l2[1].get_output_for.assert_called_with(
            l1[1].input_var)

    def test_get_output_with_single_argument_fails(self, layers, get_output):
        l1, l2, l3 = layers
        inputs, kwarg = theano.tensor.matrix(), object()
        # expected to fail: only gave one expression for two input layers
        with pytest.raises(ValueError):
            output = get_output(l3, inputs, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layers, get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1[0]).input_var = p
        type(l1[1]).input_var = p
        inputs = {l3: theano.tensor.matrix()}
        # expected: inputs[l3]
        assert get_output(l3, inputs) is inputs[l3]
        # l3.get_output_for, l2[*].get_output_for should not have been called
        assert l3.get_output_for.call_count == 0
        assert l2[0].get_output_for.call_count == 0
        assert l2[1].get_output_for.call_count == 0
        # l1[*].input_var should not have been accessed
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_no_key(self, layers, get_output):
        l1, l2, l3 = layers
        output = get_output(l3, {})
        # expected: l3.get_output_for([l2[0].get_output_for(l1[0].input_var),
        #                              l2[1].get_output_for(l1[1].input_var)])
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with([
            l2[0].get_output_for.return_value,
            l2[1].get_output_for.return_value,
            ])
        l2[0].get_output_for.assert_called_with(
            l1[0].input_var)
        l2[1].get_output_for.assert_called_with(
            l1[1].input_var)

    def test_get_output_input_is_a_mapping_to_array(self, layers, get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1[0]).input_var = p
        type(l1[1]).input_var = p
        inputs = {l3: [[1, 2, 3]]}
        output = get_output(l3, inputs)
        # expected: inputs[l3]
        assert numpy.all(output.eval() == inputs[l3])
        # l3.get_output_for, l2[*].get_output_for should not have been called
        assert l3.get_output_for.call_count == 0
        assert l2[0].get_output_for.call_count == 0
        assert l2[1].get_output_for.call_count == 0
        # l1[*].input_var should not have been accessed
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_layer(self, layers, get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1[0]).input_var = p
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {l2[0]: input_expr}
        output = get_output(l3, inputs, kwarg=kwarg)
        # expected: l3.get_output_for([input_expr,
        #                              l2[1].get_output_for(l1[1].input_var,
        #                                                   kwarg=kwarg)],
        #                              kwarg=kwarg)
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with([
            input_expr,
            l2[1].get_output_for.return_value,
            ], kwarg=kwarg)
        l2[1].get_output_for.assert_called_with(
            l1[1].input_var, kwarg=kwarg)
        # l2[0].get_output_for should not have been called
        assert l2[0].get_output_for.call_count == 0
        # l1[0].input_var should not have been accessed
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_input_layer(self, layers,
                                                           get_output):
        l1, l2, l3 = layers
        p = PropertyMock()
        type(l1[0]).input_var = p
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {l1[0]: input_expr}
        output = get_output(l3, inputs, kwarg=kwarg)
        # expected: l3.get_output_for([l2[0].get_output_for(input_expr,
        #                                                   kwarg=kwarg),
        #                              l2[1].get_output_for(l1[1].input_var,
        #                                                   kwarg=kwarg)],
        #                              kwarg=kwarg)
        assert output is l3.get_output_for.return_value
        l3.get_output_for.assert_called_with([
            l2[0].get_output_for.return_value,
            l2[1].get_output_for.return_value,
            ], kwarg=kwarg)
        l2[0].get_output_for.assert_called_with(
            input_expr, kwarg=kwarg)
        l2[1].get_output_for.assert_called_with(
            l1[1].input_var, kwarg=kwarg)
        # l1[0].input_var should not have been accessed
        assert p.call_count == 0

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.input import InputLayer
        from lasagne.layers.base import MergeLayer
        return MergeLayer([(None, 20), Mock(InputLayer((None,)))])

    def test_layer_from_shape_invalid_get_output(self, layer_from_shape,
                                                 get_output):
        layer = layer_from_shape
        with pytest.raises(ValueError):
            get_output(layer)
        with pytest.raises(ValueError):
            get_output(layer, [1, 2])
        with pytest.raises(ValueError):
            get_output(layer, {layer.input_layers[1]: [1, 2]})

    def test_layer_from_shape_valid_get_output(self, layer_from_shape,
                                               get_output):
        layer = layer_from_shape
        inputs = {layer: theano.tensor.matrix()}
        assert get_output(layer, inputs) is inputs[layer]
        inputs = {None: theano.tensor.matrix()}
        layer.get_output_for = Mock()
        assert get_output(layer, inputs) is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            [inputs[None], layer.input_layers[1].input_var])


class TestGetOutputShape_InputLayer:
    @pytest.fixture
    def get_output_shape(self):
        from lasagne.layers.helper import get_output_shape
        return get_output_shape

    @pytest.fixture
    def layer(self):
        from lasagne.layers.input import InputLayer
        return InputLayer((3, 2))

    def test_get_output_shape_without_arguments(self, layer, get_output_shape):
        assert get_output_shape(layer) == (3, 2)

    def test_get_output_shape_input_is_tuple(self, layer, get_output_shape):
        shp = (4, 5, 6)
        assert get_output_shape(layer, shp) == shp

    def test_get_output_shape_input_is_a_mapping(self, layer,
                                                 get_output_shape):
        input_shapes = {layer: (4, 5, 6)}
        assert get_output_shape(layer, input_shapes) == input_shapes[layer]


class TestGetOutputShape_Layer:
    @pytest.fixture
    def get_output_shape(self):
        from lasagne.layers.helper import get_output_shape
        return get_output_shape

    @pytest.fixture
    def layers(self):
        from lasagne.layers.base import Layer
        from lasagne.layers.input import InputLayer
        # create a mock that has the same attributes as an InputLayer instance
        l1 = Mock(InputLayer((None,)))
        # create a mock that has the same attributes as a Layer instance
        l2 = Mock(Layer(l1))
        # link it to the InputLayer mock
        l2.input_layer = l1
        # create another mock that has the same attributes as a Layer instance
        l3 = Mock(Layer(l2))
        # link it to the first mock, to get an "l1 --> l2 --> l3" chain
        l3.input_layer = l2
        return l1, l2, l3

    def test_get_output_shape_without_arguments(self, layers,
                                                get_output_shape):
        l1, l2, l3 = layers
        output_shape = get_output_shape(l3)
        # expected: l3.output_shape
        assert output_shape is l3.output_shape
        # l3.get_output_shape_for, l2.get_output_shape_for should not have been
        # called
        assert l3.get_output_shape_for.call_count == 0
        assert l2.get_output_shape_for.call_count == 0

    def test_get_output_shape_with_single_argument(self, layers,
                                                   get_output_shape):
        l1, l2, l3 = layers
        shp = (3, 4, 5)
        output_shape = get_output_shape(l3, shp)
        # expected: l3.get_output_shape_for(l2.get_output_shape_for(shp))
        assert output_shape is l3.get_output_shape_for.return_value
        l3.get_output_shape_for.assert_called_with(
            l2.get_output_shape_for.return_value)
        l2.get_output_shape_for.assert_called_with(shp)

    def test_get_output_shape_input_is_a_mapping(self, layers,
                                                 get_output_shape):
        l1, l2, l3 = layers
        input_shapes = {l3: (4, 5, 6)}
        # expected: input_shapes[l3]
        assert get_output_shape(l3, input_shapes) is input_shapes[l3]
        # l3.get_output_shape_for, l2.get_output_shape_for should not have been
        # called
        assert l3.get_output_shape_for.call_count == 0
        assert l2.get_output_shape_for.call_count == 0

    def test_get_output_shape_input_is_a_mapping_no_key(self, layers,
                                                        get_output_shape):
        l1, l2, l3 = layers
        output_shape = get_output_shape(l3, {})
        # expected: l3.output_shape
        assert output_shape is l3.output_shape
        # l3.get_output_shape_for, l2.get_output_shape_for should not have been
        # called
        assert l3.get_output_shape_for.call_count == 0
        assert l2.get_output_shape_for.call_count == 0

    def test_get_output_shape_input_is_a_mapping_for_layer(self, layers,
                                                           get_output_shape):
        l1, l2, l3 = layers
        shp = (4, 5, 6)
        input_shapes = {l2: shp}
        output_shape = get_output_shape(l3, input_shapes)
        # expected: l3.get_output_shape_for(shp)
        assert output_shape is l3.get_output_shape_for.return_value
        l3.get_output_shape_for.assert_called_with(shp)
        # l2.get_output_shape_for should not have been called
        assert l2.get_output_shape_for.call_count == 0

    def test_get_output_shape_input_is_a_mapping_for_input_layer(
            self, layers, get_output_shape):
        l1, l2, l3 = layers
        shp = (4, 5, 6)
        input_shapes = {l1: shp}
        output_shape = get_output_shape(l3, input_shapes)
        # expected: l3.get_output_shape_for(l2.get_output_shape_for(shp))
        assert output_shape is l3.get_output_shape_for.return_value
        l3.get_output_shape_for.assert_called_with(
            l2.get_output_shape_for.return_value)
        l2.get_output_shape_for.assert_called_with(shp)

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import Layer
        return Layer((None, 20))

    def test_layer_from_shape(self, layer_from_shape, get_output_shape):
        layer = layer_from_shape
        input_shapes = {layer: (4, 5, 6)}
        assert get_output_shape(layer, input_shapes) is input_shapes[layer]
        input_shapes = {None: (4, 5, 6)}
        layer.get_output_shape_for = Mock()
        assert (get_output_shape(layer, input_shapes) is
                layer.get_output_shape_for.return_value)
        layer.get_output_shape_for.assert_called_with(input_shapes[None])


class TestGetOutputShape_MergeLayer:
    @pytest.fixture
    def get_output_shape(self):
        from lasagne.layers.helper import get_output_shape
        return get_output_shape

    @pytest.fixture
    def layers(self):
        from lasagne.layers.base import Layer, MergeLayer
        from lasagne.layers.input import InputLayer
        # create two mocks of the same attributes as an InputLayer instance
        l1 = [Mock(InputLayer((None,))), Mock(InputLayer((None,)))]
        # create two mocks of the same attributes as a Layer instance
        l2 = [Mock(Layer(l1[0])), Mock(Layer(l1[1]))]
        # link them to the InputLayer mocks
        l2[0].input_layer = l1[0]
        l2[1].input_layer = l1[1]
        # create a mock that has the same attributes as a MergeLayer
        l3 = Mock(MergeLayer(l2))
        # link it to the two layer mocks, to get the following network:
        # l1[0] --> l2[0] --> l3
        # l1[1] --> l2[1] ----^
        l3.input_layers = l2
        return l1, l2, l3

    def test_get_output_shape_without_arguments(self, layers,
                                                get_output_shape):
        l1, l2, l3 = layers
        output = get_output_shape(l3)
        # expected: l3.output_shape
        assert get_output_shape(l3) is l3.output_shape
        # l3.get_output_shape_for, l2[*].get_output_shape_for should not have
        # been called
        assert l3.get_output_shape_for.call_count == 0
        assert l2[0].get_output_shape_for.call_count == 0
        assert l2[1].get_output_shape_for.call_count == 0

    def test_get_output_shape_with_single_argument_fails(self, layers,
                                                         get_output_shape):
        l1, l2, l3 = layers
        shp = (4, 5, 6)
        # expected to fail: only gave one shape tuple for two input layers
        with pytest.raises(ValueError):
            output_shape = get_output_shape(l3, shp)

    def test_get_output_shape_input_is_a_mapping(self, layers,
                                                 get_output_shape):
        l1, l2, l3 = layers
        input_shapes = {l3: (4, 5, 6)}
        # expected: input_shapes[l3]
        assert get_output_shape(l3, input_shapes) is input_shapes[l3]
        # l3.get_output_shape_for, l2[*].get_output_shape_for should not have
        # been called
        assert l3.get_output_shape_for.call_count == 0
        assert l2[0].get_output_shape_for.call_count == 0
        assert l2[1].get_output_shape_for.call_count == 0

    def test_get_output_shape_input_is_a_mapping_no_key(self, layers,
                                                        get_output_shape):
        l1, l2, l3 = layers
        output = get_output_shape(l3, {})
        # expected: l3.output_shape
        assert get_output_shape(l3) is l3.output_shape
        # l3.get_output_shape_for, l2[*].get_output_shape_for should not have
        # been called
        assert l3.get_output_shape_for.call_count == 0
        assert l2[0].get_output_shape_for.call_count == 0
        assert l2[1].get_output_shape_for.call_count == 0

    def test_get_output_shape_input_is_a_mapping_for_layer(self, layers,
                                                           get_output_shape):
        l1, l2, l3 = layers
        shp = (4, 5, 6)
        input_shapes = {l2[0]: shp}
        output = get_output_shape(l3, input_shapes)
        # expected: l3.get_output_shape_for(
        #     [shp, l2[1].get_output_shape_for(l1[1].shape)])
        assert output is l3.get_output_shape_for.return_value
        l3.get_output_shape_for.assert_called_with([
            shp, l2[1].get_output_shape_for.return_value])
        l2[1].get_output_shape_for.assert_called_with(l1[1].shape)
        # l2[0].get_output_shape_for should not have been called
        assert l2[0].get_output_shape_for.call_count == 0

    def test_get_output_shape_input_is_a_mapping_for_input_layer(
            self, layers, get_output_shape):
        l1, l2, l3 = layers
        shp = (4, 5, 6)
        input_shapes = {l1[0]: shp}
        output = get_output_shape(l3, input_shapes)
        # expected: l3.get_output_shape_for(
        #     [l2[0].get_output_shape_for(shp),
        #      l2[1].get_output_shape_for(l1[1].shape)])
        assert output is l3.get_output_shape_for.return_value
        l3.get_output_shape_for.assert_called_with([
            l2[0].get_output_shape_for.return_value,
            l2[1].get_output_shape_for.return_value,
            ])
        l2[0].get_output_shape_for.assert_called_with(shp)
        l2[1].get_output_shape_for.assert_called_with(l1[1].shape)

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.input import InputLayer
        from lasagne.layers.base import MergeLayer
        return MergeLayer([(None, 20), Mock(InputLayer((None,)))])

    def test_layer_from_shape_valid_get_output_shape(self, layer_from_shape,
                                                     get_output_shape):
        layer = layer_from_shape
        input_shapes = {layer: (4, 5, 6)}
        assert get_output_shape(layer, input_shapes) is input_shapes[layer]
        input_shapes = {None: (4, 5, 6)}
        layer.get_output_shape_for = Mock()
        assert (get_output_shape(layer, input_shapes) is
                layer.get_output_shape_for.return_value)
        layer.get_output_shape_for.assert_called_with(
            [input_shapes[None], layer.input_layers[1].shape])
