from __future__ import absolute_import
import mock
import numpy as np


def test_mse():
    from lasagne.objectives import mse

    output = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0],
        ])
    target = np.zeros((2, 4))

    result = mse(output, target)
    assert result.eval() == 0.5


def test_crossentropy():
    from lasagne.objectives import crossentropy

    output = np.array([
        [np.e ** -2],
        [np.e ** -1],
        ])
    target = np.ones((2, 4))

    result = crossentropy(output, target)
    assert result.eval() == 1.5


def test_objective():
    from lasagne.objectives import Objective

    input_layer = mock.Mock()
    loss_function = mock.Mock()
    input, target, arg1, kwarg1 = (object(),) * 4
    objective = Objective(input_layer, loss_function)
    result = objective.get_loss(input, target, arg1, kwarg1=kwarg1)

    # We expect that the input layer's `get_output` was called with
    # the `input` argument we provided, plus the extra positional and
    # keyword arguments.
    input_layer.get_output.assert_called_with(input, arg1, kwarg1=kwarg1)
    network_output = input_layer.get_output.return_value

    # The `network_output` and `target` are fed into the loss
    # function:
    loss_function.assert_called_with(network_output, target)
    assert result == loss_function.return_value


def test_objective_no_target():
    from lasagne.objectives import Objective

    input_layer = mock.Mock()
    loss_function = mock.Mock()
    input = object()
    objective = Objective(input_layer, loss_function)
    result = objective.get_loss(input)

    input_layer.get_output.assert_called_with(input)
    network_output = input_layer.get_output.return_value

    loss_function.assert_called_with(network_output, objective.target_var)
    assert result == loss_function.return_value
