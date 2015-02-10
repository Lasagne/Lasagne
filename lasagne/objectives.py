import theano
import theano.tensor as T


def mse(x, t, m=None):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values
        - m : mask; None for no mask, or an array the same shape as `t`
            that selects/weights the contribution of elements in `x` and `t`

    :returns:
        - output : the mean square error across all dimensions
    """
    if m is None:
        return T.mean((x - t) ** 2)
    else:
        return T.sum(((x - t) ** 2) * m)


def crossentropy(x, t, m=None):
    """Calculates the binary crossentropy mean across all dimentions,
    i.e.  feature dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values
        - m : mask; None for no mask, or an array the same shape as `t`
            that selects/weights the contribution of elements in `x` and `t`

    :returns:
        - output : the mean binary cross entropy across all dimensions
    """
    if m is None:
        return T.mean(T.nnet.binary_crossentropy(x, t))
    else:
        return T.sum(T.nnet.binary_crossentropy(x, t) * m)


def multinomial_nll(x, t, m=None):
    """Calculates the mean multinomial negative-log-loss

    :parameters:
        - x : (predicted) class probabilities; a theano expression resulting
            in a 2D array; samples run along axis 0, class probabilities
            along axis 1
        - t : (correct) class probabilities; a theano expression resulting in
            a 2D tensor that gives the class probabilities in its rows, OR
            a 1D integer array that gives the class index of each sample
            (the position of the 1 in the row in a 1-of-N encoding, or
            1-hot encoding),
        - m : mask; None for no mask, or a 1D array that selects/weights
            the contributions of the log-loss scores of each sample before
            they are summed

    :returns:
        - output : the mean multinomial negative log loss
    """
    if m is None:
        return T.mean(T.nnet.categorical_crossentropy(x, t))
    else:
        if m.ndim != 1:
            raise ValueError, 'mask must be a 1D array'
        return T.sum(T.nnet.categorical_crossentropy(x, t) * m)




class Objective(object):
    """
    Training objective

    The  `get_loss` method returns an expression that can be used for
    training with a gradient descent approach.
    """
    def __init__(self, input_layer, loss_function=mse):
        """
        Constructor

        :parameters:
            - input_layer : a `Layer` whose output is the networks prediction
                given its input
            - loss_function : a loss function of the form `f(x, t)` that
                returns a scalar loss given tensors that represent the
                predicted and true values as arguments..
        """
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")


    def get_loss(self, input=None, target=None, *args, **kwargs):
        """
        Get loss scalar expression

        :parameters:
            - input : (default `None`) an expression that results in the
                input data that is passed to the network
            - target : (default `None`) an expression that results in the
                desired output that the network is being trained to generate
                given the input
            - args : additional arguments passed to `input_layer`'s
                `get_output` method
            - kwargs : additional keyword arguments passed to `input_layer`'s
                `get_output` method

        :returns:
            - output : loss expressions
        """
        network_output = self.input_layer.get_output(input, *args, **kwargs)
        if target is None:
            target = self.target_var

        return self.loss_function(network_output, target)





class MaskedObjective(object):
    """
    Masked training objective

    The  `get_loss` method returns an expression that can be used for
    training with a gradient descent approach, with masking applied to weight
    the contribution of samples to the final loss.
    """
    def __init__(self, input_layer, loss_function=mse, normalize_mask=False):
        """
        Constructor

        :parameters:
            - input_layer : a `Layer` whose output is the networks prediction
                given its input
            - loss_function : a loss function of the form `f(x, t, m)` that
                returns a scalar loss given tensors that represent the
                predicted values, true values and mask as arguments.
        """
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        self.mask_var = T.matrix("mask")
        self.normalize_mask = normalize_mask

    def get_loss(self, input=None, target=None, mask=None,
                 normalize_mask=None, *args, **kwargs):
        """
        Get loss scalar expression

        :parameters:
            - input : (default `None`) an expression that results in the
                input data that is passed to the network
            - target : (default `None`) an expression that results in the
                desired output that the network is being trained to generate
                given the input
            - mask : None for no mask, or a mask that is the same shape
                as `target`/`self.target_var` - or will broadcast to that
                shape - that selects/weights the contributions of
                the samples to the final loss
            - normalize_mask : None to use the value passed to the
                constructor, or a bool to override it. If True, the mask will
                be normalized by dividing it by its sum before being applied.
            - args : additional arguments passed to `input_layer`'s
                `get_output` method
            - kwargs : additional keyword arguments passed to `input_layer`'s
                `get_output` method

        :returns:
            - output : loss expressions
        """
        network_output = self.input_layer.get_output(input, *args, **kwargs)
        if target is None:
            target = self.target_var
        if mask is None:
            mask = self.mask_var

        if normalize_mask is None:
            normalize_mask = self.normalize_mask

        if normalize_mask:
            mask = mask / T.sum(mask)

        return self.loss_function(network_output, target, mask)
