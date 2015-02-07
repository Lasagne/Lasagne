import mock
import numpy as np


def test_mse():
    from lasagne.objectives import mse

    output = np.array([
        [1.0, 0.0, 3.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0],
        ])
    target = np.zeros((2, 4))
    mask = np.array([[1.0], [0.0]])

    # Sqr-error sum = 1**2 + (-1)**2 + (-1)**2 + 3**2 = 12
    # Mean is 1.5
    result = mse(output, target)
    assert result.eval() == 1.5
    # Masked error sum is 1**2 + 3**2
    result_with_mask = mse(output, target, mask)
    assert result_with_mask.eval() == 10


def test_crossentropy():
    from lasagne.objectives import crossentropy

    output = np.array([
        [np.e ** -2]*4,
        [np.e ** -1]*4,
        ])
    target = np.ones((2, 4))
    mask = np.array([[0.0], [0.25]])

    # Cross entropy sum is (2*4) + (1*4) = 12
    # Mean is 1.5
    result = crossentropy(output, target)
    assert result.eval() == 1.5
    # Masked cross entropy sum is 1*4*0.25 = 1
    result_with_mask = crossentropy(output, target, mask)
    assert result_with_mask.eval() == 1


def test_multinomial_nll():
    from lasagne.objectives import multinomial_nll

    output = np.array([
        [1.0, 1.0-np.e**-1, np.e**-1],
        [1.0-np.e**-2, np.e**-2, 1.0],
        [1.0-np.e**-3, 1.0, np.e**-3]
        ])
    target_1hot = np.array([2,1,2])
    target_2d = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    mask_1hot = np.array([0,1,1])

    # Multinomial NLL sum is 1 + 2 + 3 = 6
    # Mean is 2
    result = multinomial_nll(output, target_1hot)
    assert result.eval() == 2
    # Multinomial NLL sum is (0*0 + 1*0 + 1*1) + (2*0 + 2*1 + 0*0)
    # + (3*0 + 0*0 + 3*1) = 6
    # Mean is 2
    result = multinomial_nll(output, target_2d)
    assert result.eval() == 2
    # Masked NLL sum is 2 + 3 = 5
    result_with_mask = multinomial_nll(output, target_1hot, mask_1hot)
    assert result_with_mask.eval() == 5
    # Masked NLL sum is 2 + 3 = 5
    result_with_mask = multinomial_nll(output, target_2d, mask_1hot)
    assert result_with_mask.eval() == 5


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


def test_masked_objective():
    from lasagne.objectives import MaskedObjective

    input_layer = mock.Mock()
    loss_function = mock.Mock()
    input, target, mask, arg1, kwarg1 = (object(),) * 5
    objective = MaskedObjective(input_layer, loss_function)
    result = objective.get_loss(input, target, mask, False, arg1, kwarg1=kwarg1)

    # We expect that the input layer's `get_output` was called with
    # the `input` argument we provided, plus the extra positional and
    # keyword arguments.
    input_layer.get_output.assert_called_with(input, arg1, kwarg1=kwarg1)
    network_output = input_layer.get_output.return_value

    # The `network_output` and `target` are fed into the loss
    # function:
    loss_function.assert_called_with(network_output, target, mask)
    assert result == loss_function.return_value
