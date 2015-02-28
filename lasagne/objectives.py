import theano.tensor as T


def mse(x, t):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values

    :returns:
        - output : the mean square error across all dimensions
    """
    return T.mean((x - t) ** 2)


def crossentropy(x, t):
    """Calculates the binary crossentropy mean across all dimentions,
    i.e.  feature dimension AND minibatch dimension.

    :parameters:
        - x : predicted values
        - t : target values

    :returns:
        - output : the mean binary cross entropy across all dimensions
    """
    return T.mean(T.nnet.binary_crossentropy(x, t))


def multinomial_nll(x, t):
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

    :returns:
        - output : the mean multinomial negative log loss
    """
    return T.mean(T.nnet.categorical_crossentropy(x, t))


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

    def get_loss(self, input=None, target=None, *args, **kwargs):
        """
        Get loss scalar expression

        :parameters:
            - input : (default `None`) an expression that results in the input
                data that is passed to the network
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
