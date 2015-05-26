import mock
import numpy as np
import theano
import pytest


class TestObjectives:
    @pytest.fixture
    def input_layer(self, value):
        from lasagne.layers import InputLayer
        shape = np.array(value).shape
        x = theano.shared(value)
        return InputLayer(shape, input_var=x)

    @pytest.fixture
    def get_loss(self, loss_function, output, target, aggregation=None):
        from lasagne.objectives import Objective
        input_layer = self.input_layer(output)
        obj = Objective(input_layer, loss_function)
        return obj.get_loss(target=target, aggregation=aggregation)

    @pytest.fixture
    def get_masked_loss(self, loss_function, output, target, mask,
                        aggregation=None):
        from lasagne.objectives import MaskedObjective
        input_layer = self.input_layer(output)
        obj = MaskedObjective(input_layer, loss_function)
        return obj.get_loss(target=target, mask=mask,
                            aggregation=aggregation)

    def test_mse(self):
        from lasagne.objectives import mse

        output = np.array([
            [1.0, 0.0, 3.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0],
            ])
        target = np.zeros((2, 4))
        mask = np.array([[1.0], [0.0]])
        mask_2d = np.array([[1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]])

        # Sqr-error sum = 1**2 + (-1)**2 + (-1)**2 + 3**2 = 12
        # Mean is 1.5
        result = self.get_loss(mse, output, target, aggregation='mean')
        assert result.eval() == 1.5
        result = self.get_loss(mse, output, target, aggregation='sum')
        assert result.eval() == 12

        # Masked error sum is 1**2 + 3**2
        result_with_mask = self.get_masked_loss(mse, output, target,
                                                mask, aggregation='sum')
        assert result_with_mask.eval() == 10
        result_with_mask = self.get_masked_loss(mse, output, target,
                                                mask_2d, aggregation='sum')
        assert result_with_mask.eval() == 10
        result_with_mask = self.get_masked_loss(mse, output, target,
                                                mask, aggregation='mean')
        assert result_with_mask.eval() == 10/8.0
        result_with_mask = self.get_masked_loss(mse, output, target,
                                                mask_2d, aggregation='mean')
        assert result_with_mask.eval() == 10/8.0
        result_with_mask = self.get_masked_loss(mse, output, target,
                                                mask, aggregation=None)
        assert result_with_mask.eval() == 10/8.0
        result_with_mask = self.get_masked_loss(mse, output, target,
                                                mask_2d, aggregation=None)
        assert result_with_mask.eval() == 10/8.0
        result_with_mask = self.get_masked_loss(mse, output, target, mask,
                                                aggregation='normalized_sum')
        assert result_with_mask.eval() == 10
        result_with_mask = self.get_masked_loss(mse, output, target, mask_2d,
                                                aggregation='normalized_sum')
        assert result_with_mask.eval() == 10/4.0

    def test_binary_crossentropy(self):
        from lasagne.objectives import binary_crossentropy

        output = np.array([
            [np.e ** -2]*4,
            [np.e ** -1]*4,
            ])
        target = np.ones((2, 4))
        mask = np.array([[0.0], [1.0]])
        mask_2d = np.array([[0.0]*4,
                            [1.0]*4])

        # Cross entropy sum is (2*4) + (1*4) = 12
        # Mean is 1.5
        result = self.get_loss(binary_crossentropy, output, target,
                               aggregation='mean')
        assert result.eval() == 1.5
        result = self.get_loss(binary_crossentropy, output, target,
                               aggregation='sum')
        assert result.eval() == 12

        # Masked cross entropy sum is 1*4*1 = 4
        result_with_mask = self.get_masked_loss(binary_crossentropy,
                                                output, target, mask,
                                                aggregation='sum')
        assert result_with_mask.eval() == 4
        result_with_mask = self.get_masked_loss(binary_crossentropy,
                                                output, target, mask_2d,
                                                aggregation='sum')
        assert result_with_mask.eval() == 4
        result_with_mask = self.get_masked_loss(binary_crossentropy,
                                                output, target, mask,
                                                aggregation='mean')
        assert result_with_mask.eval() == 1/2.0
        result_with_mask = self.get_masked_loss(binary_crossentropy,
                                                output, target, mask_2d,
                                                aggregation='mean')
        assert result_with_mask.eval() == 1/2.0
        result_with_mask = self.get_masked_loss(binary_crossentropy,
                                                output, target, mask,
                                                aggregation='normalized_sum')
        assert result_with_mask.eval() == 4
        result_with_mask = self.get_masked_loss(binary_crossentropy,
                                                output, target, mask_2d,
                                                aggregation='normalized_sum')
        assert result_with_mask.eval() == 1

    def test_categorical_crossentropy(self):
        from lasagne.objectives import categorical_crossentropy

        output = np.array([
            [1.0, 1.0-np.e**-1, np.e**-1],
            [1.0-np.e**-2, np.e**-2, 1.0],
            [1.0-np.e**-3, 1.0, np.e**-3]
            ])
        target_1hot = np.array([2, 1, 2])
        target_2d = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        mask_1hot = np.array([0, 1, 1])

        # Multinomial NLL sum is 1 + 2 + 3 = 6
        # Mean is 2
        result = self.get_loss(categorical_crossentropy, output, target_1hot,
                               aggregation='mean')
        assert result.eval() == 2
        result = self.get_loss(categorical_crossentropy, output, target_1hot,
                               aggregation='sum')
        assert result.eval() == 6
        # Multinomial NLL sum is (0*0 + 1*0 + 1*1) + (2*0 + 2*1 + 0*0)
        # + (3*0 + 0*0 + 3*1) = 6
        # Mean is 2
        result = self.get_loss(categorical_crossentropy, output, target_2d,
                               aggregation='mean')
        assert result.eval() == 2
        result = self.get_loss(categorical_crossentropy, output, target_2d,
                               aggregation='sum')
        assert result.eval() == 6

        # Masked NLL sum is 2 + 3 = 5
        result_with_mask = self.get_masked_loss(categorical_crossentropy,
                                                output, target_1hot,
                                                mask_1hot,
                                                aggregation='sum')
        assert result_with_mask.eval() == 5

        # Masked NLL sum is 2 + 3 = 5
        result_with_mask = self.get_masked_loss(categorical_crossentropy,
                                                output, target_2d, mask_1hot,
                                                aggregation='mean')
        assert abs(result_with_mask.eval() - 5.0/3.0) < 1.0e-9

        # Masked NLL sum is 2 + 3 = 5
        result_with_mask = self.get_masked_loss(categorical_crossentropy,
                                                output, target_2d, mask_1hot,
                                                aggregation='normalized_sum')
        assert result_with_mask.eval() == 5.0/2.0

    def test_objective(self):
        from lasagne.objectives import Objective
        from lasagne.layers.input import Layer, InputLayer

        input_layer = mock.Mock(InputLayer((None,)), output_shape=(None,))
        layer = mock.Mock(Layer(input_layer), output_shape=(None,))
        layer.input_layer = input_layer
        loss_function = mock.Mock()
        input, target, kwarg1 = theano.tensor.vector(), object(), object()
        objective = Objective(layer, loss_function)
        result = objective.get_loss(input, target, 'mean', kwarg1=kwarg1)

        # We expect that the layer's `get_output_for` was called with
        # the `input` argument we provided, plus the extra positional and
        # keyword arguments.
        layer.get_output_for.assert_called_with(input, kwarg1=kwarg1)
        network_output = layer.get_output_for.return_value

        # The `network_output` and `target` are fed into the loss
        # function:
        loss_function.assert_called_with(network_output, target)
        assert result == loss_function.return_value.mean.return_value

    def test_objective_no_target(self):
        from lasagne.objectives import Objective
        from lasagne.layers.input import Layer, InputLayer

        input_layer = mock.Mock(InputLayer((None,)), output_shape=(None,))
        layer = mock.Mock(Layer(input_layer), output_shape=(None,))
        layer.input_layer = input_layer
        loss_function = mock.Mock()
        input = theano.tensor.vector()
        objective = Objective(layer, loss_function)
        result = objective.get_loss(input)

        layer.get_output_for.assert_called_with(input)
        network_output = layer.get_output_for.return_value

        loss_function.assert_called_with(network_output, objective.target_var)
        assert result == loss_function.return_value.mean.return_value
