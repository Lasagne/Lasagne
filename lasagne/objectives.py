import theano
import theano.tensor as T


def mse(x, t, m):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values
        - m : mask; None for no mask, or a 1-dimensional array, that
            selects which rows in `x` and `t` should count towards the
            result

    :returns:
        - output : the mean square error across all dimensions
    """
    if m is None:
        return T.mean((x - t) ** 2)
    else:
        recip_mask_sum = T.cast(1.0 / T.sum(m), theano.config.floatX)
        return T.sum(m * (x - t) ** 2) * recip_mask_sum


def crossentropy(x, t, m):
    """Calculates the binary crossentropy mean across all dimentions,
    i.e.  feature dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values
        - m : mask; None for no mask, or a 1-dimensional array, that
            selects which rows in `x` and `t` should count towards the
            result

    :returns:
        - output : the mean binary cross entropy across all dimensions
    """
    if m is None:
        return T.mean(T.nnet.binary_crossentropy(x, t))
    else:
        recip_mask_sum = T.cast(1.0 / T.sum(m), theano.config.floatX)
        return T.sum(T.nnet.binary_crossentropy(x, t)) * recip_mask_sum


def multinomial_nll(x, t, m):
    """Calculates the mean multinomial negative-log-loss

    :parameters:
        - x : (predicted) class probabilities; a theano expression resulting
            in a 2D array; samples run along axis 0, class probabilities along
            axis 1
        - t : (correct) class probabilities; a theano expression resulting in
            a 2D tensor that gives the class probabilities in its rows, OR
            a 1D integer array that gives the class index of each sample
            (the position of the 1 in the row in a 1-of-N encoding, or
            1-hot encoding),
        - m : mask; None for no mask, or a 1-dimensional array, that
            selects which rows in `x` and `t` should count towards the
            result

    :returns:
        - output : the mean multinomial negative log loss
    """
    if m is None:
        return -T.mean(T.log(x)[T.arange(t.shape[0]), t])
    else:
        recip_mask_sum = T.cast(1.0 / T.sum(m), theano.config.floatX)
        masked_nll_sum = T.sum(T.log(x)[T.arange(t.shape[0]), t] * m)
        return -masked_nll_sum * recip_mask_sum




class Objective(object):
    """
    Training objective

    The  `get_loss` method returns an expression that can be used for training
    with a gradient descent approach.
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
        self.mask_var = None

    def get_loss(self, input=None, target=None, mask=None, *args, **kwargs):
        """
        Get loss scalar expression

        :parameters:
            - input : (default `None`) an expression that results in the input
                data that is passed to the network
            - target : (default `None`) an expression that results in the
                desired output that the network is being trained to generate
                given the input
            - mask : (default `None`) [optional] an expression that results in
                a mask. Zero values in the mask will prevent the values in
                the corresponding rows in `input` and `target` from counting
                toward the loss value that is returned
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

        return self.loss_function(network_output, target, mask)
