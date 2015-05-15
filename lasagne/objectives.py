import theano
import theano.tensor as T
from theano.tensor.nnet import binary_crossentropy, categorical_crossentropy
from lasagne.layers import get_output


def mse(x, t):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values

    :returns:
        - output : the mean square error across all dimensions
    """
    return (x - t) ** 2


class Objective(object):
    _valid_aggregation = {None, 'mean', 'sum'}

    """
    Training objective

    The  `get_loss` method returns cost expression useful for training or
    evaluating a network.
    """
    def __init__(self, input_layer, loss_function=mse, aggregation='mean'):
        """
        Constructor

        :parameters:
            - input_layer : a `Layer` whose output is the networks prediction
                given its input
            - loss_function : a loss function of the form `f(x, t)` that
                returns a scalar loss given tensors that represent the
                predicted and true values as arguments..
            - aggregation : either:
                - `'mean'` or `None` : the mean of the the elements of the
                loss will be returned
                - `'sum'` : the sum of the the elements of the loss will be
                returned
        """
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        self.aggregation = aggregation

    def get_loss(self, input=None, target=None, aggregation=None, **kwargs):
        """
        Get loss scalar expression

        :parameters:
            - input : (default `None`) an expression that results in the
                input data that is passed to the network
            - target : (default `None`) an expression that results in the
                desired output that the network is being trained to generate
                given the input
            - aggregation : None to use the value passed to the
                constructor or a value to override it
            - kwargs : additional keyword arguments passed to `input_layer`'s
                `get_output` method

        :returns:
            - output : loss expressions
        """
        network_output = get_output(self.input_layer, input, **kwargs)
        if target is None:
            target = self.target_var
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        if aggregation is None:
            aggregation = self.aggregation

        losses = self.loss_function(network_output, target)

        if aggregation is None or aggregation == 'mean':
            return losses.mean()
        elif aggregation == 'sum':
            return losses.sum()
        else:
            raise RuntimeError('This should have been caught earlier')


class MaskedObjective(object):
    _valid_aggregation = {None, 'sum', 'mean', 'normalized_sum'}

    """
    Masked training objective

    The  `get_loss` method returns an expression that can be used for
    training with a gradient descent approach, with masking applied to weight
    the contribution of samples to the final loss.
    """
    def __init__(self, input_layer, loss_function=mse, aggregation='mean'):
        """
        Constructor

        :parameters:
            - input_layer : a `Layer` whose output is the networks prediction
                given its input
            - loss_function : a loss function of the form `f(x, t, m)` that
                returns a scalar loss given tensors that represent the
                predicted values, true values and mask as arguments.
            - aggregation : either:
                - `None` or `'mean'` : the elements of the loss will be
                multiplied by the mask and the mean returned
                - `'sum'` : the elements of the loss will be multiplied by
                the mask and the sum returned
                - `'normalized_sum'` : the elements of the loss will be
                multiplied by the mask, summed and divided by the sum of
                the mask
        """
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        self.mask_var = T.matrix("mask")
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             '\'normalized_sum\' or None,'
                             ' not {0}'.format(aggregation))
        self.aggregation = aggregation

    def get_loss(self, input=None, target=None, mask=None,
                 aggregation=None, **kwargs):
        """
        Get loss scalar expression

        :parameters:
            - input : (default `None`) an expression that results in the
                input data that is passed to the network
            - target : (default `None`) an expression that results in the
                desired output that the network is being trained to generate
                given the input
            - mask : None for no mask, or a soft mask that is the same shape
                as - or broadcast-able to the shape of - the result of
                applying the loss function. It selects/weights the
                contributions of the resulting loss values
            - aggregation : None to use the value passed to the
                constructor or a value to override it
            - kwargs : additional keyword arguments passed to `input_layer`'s
                `get_output` method

        :returns:
            - output : loss expressions
        """
        network_output = get_output(self.input_layer, input, **kwargs)
        if target is None:
            target = self.target_var
        if mask is None:
            mask = self.mask_var

        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             '\'normalized_sum\' or None, '
                             'not {0}'.format(aggregation))

        # Get aggregation value passed to constructor if None
        if aggregation is None:
            aggregation = self.aggregation

        masked_losses = self.loss_function(network_output, target) * mask

        if aggregation is None or aggregation == 'mean':
            return masked_losses.mean()
        elif aggregation == 'sum':
            return masked_losses.sum()
        elif aggregation == 'normalized_sum':
            return masked_losses.sum() / mask.sum()
        else:
            raise RuntimeError('This should have been caught earlier')
