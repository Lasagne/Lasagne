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
        assert get_all_layers(l6) == [l1, l2, l3, l4, l5, l6]
        assert get_all_layers([l4, l6]) == [l4, l1, l2, l3, l5, l6]
        assert get_all_layers([l5, l6]) == [l4, l5, l1, l2, l3, l6]
        assert get_all_layers([l4, l2, l5, l6]) == [l4, l1, l2, l5, l3, l6]
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
        assert get_all_layers(l3) == [l1, l2, l3]
        assert get_all_layers(l4) == [l1, l4]
        assert get_all_layers([l3, l4]) == [l1, l2, l3, l4]
        assert get_all_layers([l4, l3]) == [l1, l4, l2, l3]
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
        assert get_all_layers(l5) == [l1, l2, l3, l4, l5]
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
    def layer(self):
        from lasagne.layers.base import Layer
        from lasagne.layers.input import InputLayer
        # create a mock that has the same attributes as an InputLayer instance
        input_layer = Mock(InputLayer((None,)))
        # create a mock that has the same attributes as a Layer instance
        layer1 = Mock(Layer(input_layer))
        # link it to the InputLayer mock
        layer1.input_layer = input_layer
        # create another mock that has the same attributes as a Layer instance
        layer2 = Mock(Layer(layer1))
        # link it to the first mock, to get an "input -> layer -> layer" chain
        layer2.input_layer = layer1
        return layer2

    def test_get_output_without_arguments(self, layer, get_output):
        output = get_output(layer)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output_for.return_value)
        layer.input_layer.get_output_for.assert_called_with(
            layer.input_layer.input_layer.input_var)

    def test_get_output_with_single_argument(self, layer, get_output):
        inputs, kwarg = theano.tensor.matrix(), object()
        output = get_output(layer, inputs, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output_for.return_value, kwarg=kwarg)
        layer.input_layer.get_output_for.assert_called_with(
            inputs, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer, get_output):
        p = PropertyMock()
        type(layer.input_layer.input_layer).input_var = p
        inputs = {layer: theano.tensor.matrix()}
        assert get_output(layer, inputs) is inputs[layer]
        assert layer.get_output_for.call_count == 0
        assert layer.input_layer.get_output_for.call_count == 0
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_no_key(self, layer, get_output):
        output = get_output(layer, {})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer, get_output):
        p = PropertyMock()
        type(layer.input_layer.input_layer).input_var = p
        inputs = {layer: [[1, 2, 3]]}
        output = get_output(layer, inputs)
        assert numpy.all(output.eval() == inputs[layer])
        assert layer.get_output_for.call_count == 0
        assert layer.input_layer.get_output_for.call_count == 0
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_layer(self, layer, get_output):
        p = PropertyMock()
        type(layer.input_layer.input_layer).input_var = p
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {layer.input_layer: input_expr}
        output = get_output(layer, inputs, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(input_expr, kwarg=kwarg)
        assert layer.input_layer.get_output_for.call_count == 0
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_input_layer(self, layer,
                                                           get_output):
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {layer.input_layer.input_layer: input_expr}
        output = get_output(layer, inputs, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output_for.return_value, kwarg=kwarg)
        layer.input_layer.get_output_for.assert_called_with(
            input_expr, kwarg=kwarg)

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


class TestGetOutput_MultipleInputsLayer:
    @pytest.fixture
    def get_output(self):
        from lasagne.layers.helper import get_output
        return get_output

    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import Layer, MultipleInputsLayer
        from lasagne.layers.input import InputLayer
        # create two mocks of the same attributes as an InputLayer instance
        input_layers = [Mock(InputLayer((None,))), Mock(InputLayer((None,)))]
        # create two mocks of the same attributes as a Layer instance
        layers = [Mock(Layer(input_layers[0])), Mock(Layer(input_layers[1]))]
        # link them to the InputLayer mocks
        layers[0].input_layer = input_layers[0]
        layers[1].input_layer = input_layers[1]
        # create a mock that has the same attributes as a MultipleInputsLayer
        layer = Mock(MultipleInputsLayer(input_layers))
        # link it to the two "input -> layer" mocks
        layer.input_layers = layers
        return layer

    def test_get_output_without_arguments(self, layer, get_output):
        output = get_output(layer)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            layer.input_layers[0].get_output_for.return_value,
            layer.input_layers[1].get_output_for.return_value,
            ])
        layer.input_layers[0].get_output_for.assert_called_with(
            layer.input_layers[0].input_layer.input_var)
        layer.input_layers[1].get_output_for.assert_called_with(
            layer.input_layers[1].input_layer.input_var)

    def test_get_output_with_single_argument_fails(self, layer, get_output):
        inputs, kwarg = theano.tensor.matrix(), object()
        with pytest.raises(ValueError):
            output = get_output(layer, inputs, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer, get_output):
        p = PropertyMock()
        type(layer.input_layers[0].input_layer).input_var = p
        type(layer.input_layers[1].input_layer).input_var = p
        inputs = {layer: theano.tensor.matrix()}
        assert get_output(layer, inputs) is inputs[layer]
        assert layer.get_output_for.call_count == 0
        assert layer.input_layers[0].get_output_for.call_count == 0
        assert layer.input_layers[1].get_output_for.call_count == 0
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_no_key(self, layer, get_output):
        output = get_output(layer, {})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer, get_output):
        p = PropertyMock()
        type(layer.input_layers[0].input_layer).input_var = p
        type(layer.input_layers[1].input_layer).input_var = p
        inputs = {layer: [[1, 2, 3]]}
        output = get_output(layer, inputs)
        assert numpy.all(output.eval() == inputs[layer])
        assert layer.get_output_for.call_count == 0
        assert layer.input_layers[0].get_output_for.call_count == 0
        assert layer.input_layers[1].get_output_for.call_count == 0
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_layer(self, layer, get_output):
        p = PropertyMock()
        type(layer.input_layers[0].input_layer).input_var = p
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {layer.input_layers[0]: input_expr}
        output = get_output(layer, inputs, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            input_expr,
            layer.input_layers[1].get_output_for.return_value,
            ], kwarg=kwarg)
        assert layer.input_layers[0].get_output_for.call_count == 0
        layer.input_layers[1].get_output_for.assert_called_with(
            layer.input_layers[1].input_layer.input_var, kwarg=kwarg)
        assert p.call_count == 0

    def test_get_output_input_is_a_mapping_for_input_layer(self, layer,
                                                           get_output):
        input_expr, kwarg = theano.tensor.matrix(), object()
        inputs = {layer.input_layers[0].input_layer: input_expr}
        output = get_output(layer, inputs, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            layer.input_layers[0].get_output_for.return_value,
            layer.input_layers[1].get_output_for.return_value,
            ], kwarg=kwarg)
        layer.input_layers[0].get_output_for.assert_called_with(
            input_expr, kwarg=kwarg)
        layer.input_layers[1].get_output_for.assert_called_with(
            layer.input_layers[1].input_layer.input_var, kwarg=kwarg)

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.input import InputLayer
        from lasagne.layers.base import MultipleInputsLayer
        return MultipleInputsLayer([(None, 20), Mock(InputLayer((None,)))])

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
